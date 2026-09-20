from datetime import date, datetime
from typing import Any, Optional


def _parse_date(value: Any) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _parse_year(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        year = int(value)
        return year if 1000 <= year <= 9999 else None
    except (TypeError, ValueError):
        parsed = _parse_date(value)
        return parsed.year if parsed else None


def normalize_legal_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return canonical temporal/version fields without changing source text."""
    normalized = dict(metadata)
    decision_date = _parse_date(normalized.get("decision_date"))
    if decision_date is None:
        decision_date = _parse_date(normalized.get("judgment_date") or normalized.get("judgement_date"))
    decision_year = _parse_year(normalized.get("decision_year"))
    if decision_year is None and decision_date is not None:
        decision_year = decision_date.year
    effective_date = _parse_date(normalized.get("effective_date") or normalized.get("valid_from"))
    repeal_date = _parse_date(normalized.get("repeal_date") or normalized.get("valid_to"))

    if decision_date is not None:
        normalized["decision_date"] = decision_date.isoformat()
    if decision_year is not None:
        normalized["decision_year"] = decision_year
    if effective_date is not None:
        normalized["effective_date"] = effective_date.isoformat()
    if repeal_date is not None:
        normalized["repeal_date"] = repeal_date.isoformat()

    version = normalized.get("statute_version") or normalized.get("version")
    if version is not None and str(version).strip():
        normalized["statute_version"] = str(version).strip()
    return normalized


def extract_temporal_metadata(record: dict[str, Any], document_id: str) -> dict[str, Any]:
    """Preserve supported legal date/version fields from an input record."""
    candidates = {
        "decision_date": ("decision_date", "judgment_date", "judgement_date"),
        "decision_year": ("decision_year", "year"),
        "effective_date": ("effective_date", "valid_from"),
        "repeal_date": ("repeal_date", "valid_to"),
        "statute_version": ("statute_version", "version"),
    }
    metadata: dict[str, Any] = {"document_id": document_id}
    for canonical, keys in candidates.items():
        for key in keys:
            value = record.get(key)
            if value not in (None, ""):
                metadata[canonical] = value
                break
    return normalize_legal_metadata(metadata)


def matches_temporal_constraints(
    metadata: dict[str, Any],
    decision_year: Optional[int] = None,
    as_of: Optional[str] = None,
    statute_version: Optional[str] = None,
) -> bool:
    """Return whether metadata satisfies every supplied legal temporal constraint."""
    normalized = normalize_legal_metadata(metadata)

    if decision_year is not None:
        if normalized.get("decision_year") != int(decision_year):
            return False

    if as_of is not None:
        requested_date = _parse_date(as_of)
        if requested_date is None:
            raise ValueError("as_of must be a parseable date")
        decision_date = _parse_date(normalized.get("decision_date"))
        if decision_date is None or decision_date > requested_date:
            return False
        effective_date = _parse_date(normalized.get("effective_date"))
        if effective_date is not None and requested_date < effective_date:
            return False
        repeal_date = _parse_date(normalized.get("repeal_date"))
        if repeal_date is not None and requested_date > repeal_date:
            return False

    if statute_version is not None:
        indexed_version = normalized.get("statute_version")
        if indexed_version is None or str(indexed_version).casefold() != str(statute_version).casefold():
            return False

    return True
