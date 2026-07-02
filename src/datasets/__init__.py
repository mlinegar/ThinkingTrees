"""
Dataset registry.

Datasets describe where documents come from (manifestos, JSONL, etc.).
"""

if __name__ == "datasets":
    import importlib.machinery
    import importlib.util
    import sys
    from pathlib import Path

    _here = Path(__file__).resolve()
    _search_paths = [
        path
        for path in sys.path
        if "site-packages" in path or "dist-packages" in path
    ]
    _spec = importlib.machinery.PathFinder.find_spec("datasets", _search_paths)
    if _spec is not None and _spec.loader is not None and _spec.origin:
        if Path(_spec.origin).resolve() != _here:
            _module = importlib.util.module_from_spec(_spec)
            sys.modules[__name__] = _module
            _spec.loader.exec_module(_module)
else:
    from .base import (
        DatasetInfo,
        DatasetPlugin,
        DatasetRegistry,
        register_dataset,
    )

    from .manifesto import ManifestoDataset
    from .jsonl import JSONLDataset
    from .pdf import PDFDataset


    def get_dataset(name: str, **kwargs):
        """Get a dataset instance by name."""
        return DatasetRegistry.get(name, **kwargs)


    def list_datasets():
        """List all registered datasets."""
        return DatasetRegistry.list_datasets()


    __all__ = [
        "DatasetInfo",
        "DatasetPlugin",
        "DatasetRegistry",
        "register_dataset",
        "get_dataset",
        "list_datasets",
        "ManifestoDataset",
        "JSONLDataset",
        "PDFDataset",
    ]
