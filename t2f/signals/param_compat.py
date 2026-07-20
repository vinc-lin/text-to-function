from ..types import FunctionCard, LexFeatures


def _has_unit(card: FunctionCard, unit: str) -> bool:
    return any(p.unit == unit for p in card.params)


def _has_bool(card: FunctionCard) -> bool:
    return any(p.type == "boolean" for p in card.params)


def _has_position_enum(card: FunctionCard) -> bool:
    pos = {"driver", "passenger", "rear", "all", "left", "right"}
    return any(p.type == "enum" and p.enum and (set(p.enum) & pos) for p in card.params)


def param_compat_score(features: LexFeatures, card: FunctionCard) -> float:
    checks: list[float] = []
    if features.temperatures:
        checks.append(1.0 if _has_unit(card, "celsius") else 0.0)
    if features.percentages:
        checks.append(1.0 if _has_unit(card, "percent") else 0.0)
    if features.levels:
        checks.append(1.0 if _has_unit(card, "level") else 0.0)
    if features.positions:
        checks.append(1.0 if _has_position_enum(card) else 0.0)
    if features.on_off is not None:
        checks.append(1.0 if _has_bool(card) or not card.required_params else 0.0)
    if not checks:
        return 0.0
    return sum(checks) / len(checks)
