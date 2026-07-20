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

---

# Evaluation datasets (Task 19)

Reproducibility record for the gold + silver eval sets under `data/eval/`.
Loaded/validated via `eval.dataset.load_dataset` and
`eval.dataset.validate_against_catalog`. Row schema (JSONL, UTF-8, Chinese
written directly with `ensure_ascii=false` semantics):

```json
{"utterance": "副驾这边冷，帮我升到26度", "expected_functions": ["set_temperature"],
 "expected_params": {"set_temperature": {"temperature": 26, "position": "passenger"}},
 "type": "single", "split": "dev"}
```

`type ∈ {single, multi_intent, ood, ambiguous}`. `multi_intent` →
`expected_functions` has ≥2 names; `ood` → `expected_functions` is `[]`
(no `expected_params`); `ambiguous` → the utterance under-specifies which
function/params (`expected_functions` is either a single best-guess or `[]`).

## Gold set (`data/eval/gold.jsonl`) — 312 rows, hand-authored & verified

Composition:

- **192 single** — every one of the 92 catalog functions appears in **≥2**
  single-intent rows (the two-slot-per-function coverage floor is 184; the
  extra 8 go to high-frequency functions: `set_temperature`, `set_fan_speed`,
  `set_volume`, `play_music`, `navigate_to`, `open_window`, `make_call`,
  `set_ac_power`).
- **48 multi_intent** — natural compound commands (「打开车窗，再把空调调到25度」),
  each with ≥2 distinct functions, connectors 然后/再/顺便/接着.
- **48 ood** — split between chitchat / general knowledge (今天天气怎么样,
  讲个笑话, 一公里等于多少米) and **unsupported vehicle requests whose function is
  NOT in the catalog** (查一下胎压, 打开行车记录仪, 启动自动驾驶, 把电子手刹拉起来,
  设个明早七点的闹钟). All have `expected_functions: []`.
- **24 ambiguous** — under-specified utterances (调一下温度 with no value/zone,
  打开 with no object, 座椅调一下, 凉快点). About half carry a single best-guess
  function; the truly unclear ones use `[]`.

`split`: deterministic **≈40% dev / 60% test** (128 dev / 184 test, 0.41),
assigned per type so both splits span all four types (indices 0,1 of every
group of 5 → dev).

**Params policy.** `expected_params` includes ONLY values inferable from the
utterance text, with catalog-correct values: positions from
`[driver, passenger, rear, all, left, right]` (主驾→driver, 副驾→passenger,
后排→rear, 全车→all, 后排左/右→left/right); temperatures/frequencies as numbers;
levels/volume/index/percent as integers; on-off direction as booleans
(打开/开→true, 关闭/关→false; 息屏 is_off→true; 折叠 fold_mirror.enabled→true;
放倒 fold_rear_seat.enabled→true). Rows whose function's required param is not
stated (e.g. 「打开座椅通风」 with no level) simply omit that param.

### CRITICAL anti-leakage rule (gold only)

The gold set measures **generalization**, so gold utterances are **NOVEL
paraphrases** — never verbatim copies of a card's prototype `utterances` (which
retrieval matches against) nor of its `aliases`. They deliberately differ in
wording (more colloquial/spoken), numbers, positions, and synonyms from the
card prototypes. Verified programmatically: **0** gold utterances match any of
the 1148 catalog prototype-utterance/alias strings.

## Silver set (`data/eval/silver.jsonl`) — 1184 rows, generated weak labels

Lighter-effort, not hand-verified, no `split` field. The silver set MAY reuse
card material as weak labels (the anti-leakage rule does not apply to it):

- **736 single** — every card utterance (92 × 8) taken verbatim, weak-labelled
  with its owning function.
- **368 single (varied)** — every other card utterance prepended with a spoken
  filler (帮我/麻烦/那个/请/你好/我想).
- **80 multi_intent** — two card utterances from different functions joined with
  a connector (然后/再/顺便/接着), labelled with both functions.

Silver carries no `expected_params` (weak labels only).

## How the gold rows were authored & verified

1. **Extracted the real schema.** `t2f.cards.load_catalog('data/catalog')` gives
   the authoritative param names, types, units, ranges and enum values per
   function; the generator reads these directly rather than hard-coding.
2. **Hand-authored** all 312 gold rows (utterance + functions + inferable
   params + type) as Python data, deliberately paraphrasing away from card
   prototypes.
3. **Self-checked params against the extracted schema** before writing: unknown
   param names, out-of-enum values, wrong scalar type, out-of-range numbers, and
   non-canonical position values all hard-fail generation.
4. **Validated** with `validate_against_catalog` → `[]` problems on both gold
   and silver; ran `tests/test_dataset.py` and the full suite (all pass).
