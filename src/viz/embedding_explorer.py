"""UMAP 3D embedding visualization for RAG debugging.

Projects document embeddings and query vectors into 3D space using UMAP,
then renders an interactive Plotly scatter plot. Helps identify:
  - Clusters of semantically similar documents
  - Query-document proximity (are queries near their relevant docs?)
  - Coverage gaps (areas of the corpus with no neighboring queries)
  - Noise clusters (chunks that lack meaningful content)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


@dataclass
class EmbeddingExplorer:
    """Interactive 3D UMAP visualization of RAG embeddings.

    Usage:
        explorer = EmbeddingExplorer()
        explorer.add_documents(chunk_texts, chunk_embeddings, labels=doc_types)
        explorer.add_queries(query_texts, query_embeddings)
        fig = explorer.plot_3d()
        fig.show()
    """

    n_neighbors: int = 15
    min_dist: float = 0.1
    n_components: int = 3
    metric: str = "cosine"

    _doc_embeddings: list[list[float]] = field(default_factory=list, init=False)
    _doc_texts: list[str] = field(default_factory=list, init=False)
    _doc_labels: list[str] = field(default_factory=list, init=False)
    _query_embeddings: list[list[float]] = field(default_factory=list, init=False)
    _query_texts: list[str] = field(default_factory=list, init=False)

    def add_documents(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        labels: list[str] | None = None,
    ) -> None:
        """Add document chunk embeddings for visualization."""
        self._doc_embeddings.extend(embeddings)
        self._doc_texts.extend(texts)
        if labels:
            self._doc_labels.extend(labels)
        else:
            self._doc_labels.extend(["document"] * len(texts))

    def add_queries(
        self,
        texts: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """Add query embeddings for visualization."""
        self._query_embeddings.extend(embeddings)
        self._query_texts.extend(texts)

    def compute_umap(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Run UMAP dimensionality reduction.

        Returns:
            Tuple of (document_coords, query_coords). query_coords is None
            if no queries were added.
        """
        import umap

        all_embeddings = np.array(self._doc_embeddings)
        has_queries = len(self._query_embeddings) > 0

        if has_queries:
            query_embeddings = np.array(self._query_embeddings)
            combined = np.vstack([all_embeddings, query_embeddings])
        else:
            combined = all_embeddings

        reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            n_components=self.n_components,
            metric=self.metric,
        )
        projected = reducer.fit_transform(combined)

        n_docs = len(self._doc_embeddings)
        doc_coords = projected[:n_docs]
        query_coords = projected[n_docs:] if has_queries else None

        return doc_coords, query_coords

    def plot_3d(self, title: str = "RAG embedding space") -> go.Figure:
        """Create an interactive 3D scatter plot.

        Documents are colored by their label (doc_type).
        Queries are shown as larger, distinctly colored markers.
        """
        doc_coords, query_coords = self.compute_umap()

        fig = go.Figure()

        # Document points
        df_docs = pd.DataFrame({
            "x": doc_coords[:, 0],
            "y": doc_coords[:, 1],
            "z": doc_coords[:, 2],
            "text": [t[:80] + "..." if len(t) > 80 else t for t in self._doc_texts],
            "label": self._doc_labels,
        })

        for label in df_docs["label"].unique():
            subset = df_docs[df_docs["label"] == label]
            fig.add_trace(go.Scatter3d(
                x=subset["x"],
                y=subset["y"],
                z=subset["z"],
                mode="markers",
                marker=dict(size=3, opacity=0.6),
                text=subset["text"],
                hovertemplate="%{text}<extra>" + label + "</extra>",
                name=label,
            ))

        # Query points
        if query_coords is not None:
            fig.add_trace(go.Scatter3d(
                x=query_coords[:, 0],
                y=query_coords[:, 1],
                z=query_coords[:, 2],
                mode="markers",
                marker=dict(size=8, symbol="diamond", color="red", opacity=0.9),
                text=[t[:80] for t in self._query_texts],
                hovertemplate="%{text}<extra>query</extra>",
                name="queries",
            ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2",
                zaxis_title="UMAP 3",
            ),
            width=900,
            height=700,
            showlegend=True,
        )

        return fig

    def save_html(self, path: str | Path, title: str = "RAG embedding space") -> None:
        """Render the 3D plot and save as an interactive HTML file."""
        fig = self.plot_3d(title)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(path))
        print(f"Saved interactive visualization to {path}")
