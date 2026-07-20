# Catalog generation notes

Reproducibility record for the production function catalog under `data/catalog/`.

## Summary

- **92 functions across 10 domains** (climate 12, window 8, seat 12, media 12,
  light 10, door 8, navigation 8, phone 6, misc 8, display 8).
- Authored via **LLM-assisted drafting + manual curation** against the Task-3
  card schema. Every card was loaded through `t2f.cards.load_catalog` after
  authoring to catch schema/YAML errors immediately.

## How the utterances were produced

1. **Seed the frame per function.** For each function we fixed the canonical
   English `name`, a natural-Chinese `description`, and the parameter list with
   the pipeline-mandated units (`celsius` / `percent` / `level`) and the
   position enum drawn from `[driver, passenger, rear, all, left, right]`.

2. **Generate candidate utterances (LLM-assisted).** We prompted the model to
   produce 10-14 in-vehicle Chinese commands per function, explicitly mixing:
   - **formal / written** phrasings (e.g. 「设置目的地为西湖」),
   - **colloquial / spoken** phrasings (e.g. 「有点闷，温度降到19度」),
   - variants that carry **positions** (主驾/副驾/后排/全车), **numbers**,
     **units** (度/%/档) where the function takes such a parameter.

   Generation prompt (paraphrased, applied per function):

   > 你是中文车载语音助手的语料作者。针对功能「{name} — {description}」，
   > 写 10-14 条中国司机在车里会说的真实指令。要求：口语和书面混合；在涉及
   > 温区、数字、单位（度/百分比/档位）的功能里自然带上这些信息；每条独立成句，
   > 不要编号；只输出中文指令。

3. **Curate down to 8 per function.** From the candidates we manually kept 8
   utterances that were (a) natural, (b) non-duplicative in surface form, and
   (c) collectively covered both the "bare" command and parameterized variants.

4. **Aliases** (4-10 each): common Chinese nicknames / synonyms for the feature
   (e.g. 座椅加热 → 座椅加温 / 屁股加热 / 座垫加热), hand-curated.

5. **Hard negatives** (1-3 each): pulled from **confusable sibling functions**
   so retrieval/hybrid-scoring is stressed. Deliberate confusable clusters:
   - `set_temperature` vs `set_fan_speed` vs `set_seat_heating`
     (all "调到N档/N度"-shaped).
   - `set_volume` vs `set_fan_speed` (both "调到3档").
   - `open_window` vs `open_sunroof` vs `open_trunk`.
   - `set_front_defrost` vs `set_rear_defrost`; `set_front_fog_light` vs
     `set_rear_fog_light`; `lock_doors` vs `unlock_doors`; `next_track` vs
     `previous_track`; `set_window_child_lock` vs `set_door_child_lock`.
   Each sibling's phrasing is planted in the other's `hard_negatives`.

6. **Response templates**: Chinese, referencing only params that exist on the
   card. `{position}` is auto-localized by the renderer (driver→主驾, …), so
   templates read naturally whether or not a position is filled. Raw English
   enum values (mode/source/color) are kept out of the rendered text; those
   templates use a neutral confirmation ("已切换…").

## YAML landmine avoided

PyYAML `safe_load` (YAML 1.1) parses bare `on/off/yes/no/true/false` as
booleans. No param `name`, enum value, or unquoted string uses those bare
tokens. On/off semantics use a boolean param named `enabled` (or `is_open` /
`is_off` / `is_on`), never a param literally named `on`.

## Verification performed

- `load_catalog('data/catalog')` → 92 cards, 10 domains, no duplicate names.
- All param names are Python `str` (no accidental bool coercion).
- All `position` enums are subsets of the allowed position set.
- No enum value is a non-string; all response-template placeholders reference
  existing param names (or `position`).
- Per-card: ≥6 utterances (all have 8), 4-10 aliases, 1-3 hard_negatives,
  non-empty description and response_template.
- `pytest tests/test_catalog_quality.py` and the full suite pass.
