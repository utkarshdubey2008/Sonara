"""Serialization, deserialization, and schema export for SMR documents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from sonara.representation.schema import SMRDocument


class SMRSerializer:
    """Handles JSON serialization and schema export for SMR."""

    @staticmethod
    def to_dict(document: SMRDocument) -> Dict[str, Any]:
        """Convert SMRDocument to a Python dictionary."""
        return document.model_dump()

    @staticmethod
    def to_json(document: SMRDocument, indent: int = 2) -> str:
        """Serialize SMRDocument to formatted JSON string."""
        return document.model_dump_json(indent=indent)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> SMRDocument:
        """Parse dictionary into validated SMRDocument."""
        return SMRDocument.model_validate(data)

    @staticmethod
    def from_json(json_str: str) -> SMRDocument:
        """Parse JSON string into validated SMRDocument."""
        return SMRDocument.model_validate_json(json_str)

    @classmethod
    def save_json(cls, document: SMRDocument, file_path: str | Path, indent: int = 2) -> Path:
        """
        Save SMRDocument to a JSON file.

        Args:
            document: SMRDocument instance.
            file_path: Output file path.
            indent: JSON indentation.

        Returns:
            Resolved Path of the saved file.
        """
        path = Path(file_path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(cls.to_json(document, indent=indent))
        return path

    @classmethod
    def load_json(cls, file_path: str | Path) -> SMRDocument:
        """
        Load and validate SMRDocument from a JSON file.

        Args:
            file_path: Path to the JSON file.

        Returns:
            SMRDocument instance.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"SMR file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return cls.from_json(content)

    @staticmethod
    def export_json_schema() -> Dict[str, Any]:
        """Export the formal JSON Schema for SMRDocument."""
        return SMRDocument.model_json_schema()
