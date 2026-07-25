import pytest
pytestmark = pytest.mark.model

def _real_pipe():
    from t2f.config import Config
    from t2f.cards import load_catalog, load_ood_prototypes
    from t2f.embed import TransformersEmbedder
    from eval import arms
    from t2f.llm.client import TransformersXGrammarClient
    cfg = Config.load("config.yaml")
    cards = load_catalog("data/catalog")
    ood = load_ood_prototypes(cfg.ood_prototypes)
    emb = TransformersEmbedder(cfg.model_id, mrl_dim=cfg.mrl_dim)
    client = TransformersXGrammarClient(cfg.llm["model_id"], cfg.llm.get("max_new_tokens", 128))
    pipe = arms.build_arm_c_llm(cards, emb, cfg, client, ood_texts=ood)
    return pipe

def test_canonical_multi_intent():
    pipe = _real_pipe()
    pipe.state.reset({"set_window_position/driver": 30})
    rr = pipe.route("后排小孩老去按车窗，把车窗锁打开。然后主驾这边窗户再开一点，天窗开到一半。")
    executed = {a.function for a in rr.plan.actions if a.status == "executed"}
    # all three real actions execute; the context clause is suppressed
    assert "set_window_child_lock" in executed
    assert "set_sunroof_position" in executed
    assert "set_window_position" in executed
    assert all("后排小孩" not in a.span for a in rr.plan.actions)
    by_fn = {a.function: a for a in rr.plan.actions if a.status == "executed"}
    # child lock enabled; sunroof to 50% (一半); relative window resolved from state 30 + step10 = 40
    assert by_fn["set_window_child_lock"].parameters.get("enabled") is True
    assert by_fn["set_sunroof_position"].parameters.get("percent") == 50
    assert by_fn["set_window_position"].parameters.get("percent") == 40
    # the spoken reply: three confirmations, sentence-joined, narration absent
    assert rr.reply == ("已为您调整车窗儿童锁状态。"
                        "已将主驾车窗开度调整到40%。"
                        "已将天窗开度调整到50%。")
    assert "后排小孩" not in rr.reply
