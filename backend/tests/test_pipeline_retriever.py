import pytest

from src import pipeline, retriever


@pytest.mark.parametrize(
    "query,expected",
    [
        ("xin chao", True),
        ("chao ban", True),
        ("hello", True),
        ("tai nghe sony", False),
    ],
)
def test_off_topic_detection(query, expected):
    q = query.lower().strip()
    got = any(g in q for g in pipeline.OFF_TOPIC)
    assert got is expected


def test_strip_greetings_keeps_intent_words():
    cleaned = pipeline._strip_greetings("xin chao ban can laptop gaming")
    assert "xin chao" not in cleaned
    assert "laptop" in cleaned


@pytest.mark.parametrize(
    "query,expected",
    [
        ("tai nghe sony duoi 2 trieu", True),
        ("dien thoai samsung", True),
        ("xin chao ban", False),
    ],
)
def test_has_product_intent(query, expected):
    assert pipeline._has_product_intent(query) is expected


def test_fallback_out_of_scope_laptop_message():
    msg = pipeline._fallback_out_of_scope("laptop gaming rtx 4050 duoi 20 trieu")
    assert "laptop gaming" in msg.lower()
    assert "rtx 4050" in msg.lower()


def test_domain_supported_laptop_false_for_accessories():
    chunks = [
        {"product_name": "Chuot gaming RGB 6400 DPI"},
        {"product_name": "Ban phim co gaming"},
    ]
    assert pipeline._is_domain_supported("laptop gaming rtx 4050", chunks) is False


def test_domain_supported_laptop_true_for_laptop_terms():
    chunks = [
        {"product_name": "Laptop Asus Vivobook RTX 4050"},
    ]
    assert pipeline._is_domain_supported("laptop gaming rtx 4050", chunks) is True


def test_hard_constraint_digits_match():
    sources = [{"product_name": "Laptop RTX 4050 gaming"}]
    assert pipeline._hard_constraint_digits_match("laptop rtx 4050", sources) is True
    assert pipeline._hard_constraint_digits_match("laptop rtx 4070", sources) is False


def test_extract_price_range_under_3_million():
    price_range = retriever.extract_price_range("tai nghe sony duoi 3 trieu")
    assert price_range == (0, 3000000.0)


def test_extract_price_range_around_2_million():
    pmin, pmax = retriever.extract_price_range("dien thoai tam 2 trieu")
    assert pmin < 2000000 < pmax


def test_compute_keyword_score_rewards_brand_and_penalizes_noise():
    keywords = {"sony", "tai", "nghe"}
    price_range = (0, 3000000)

    good = {
        "product_name": "Tai nghe Sony WI-C100",
        "document": "Tai nghe bluetooth sony pin tot",
        "price": 1200000,
    }
    noisy = {
        "product_name": "Gau bong deo tai nghe",
        "document": "do choi gau bong",
        "price": 1200000,
    }

    good_score = retriever.compute_keyword_score(good, keywords, price_range)
    noisy_score = retriever.compute_keyword_score(noisy, keywords, price_range)
    assert good_score > noisy_score


def test_qdrant_vector_scoring_semantics(monkeypatch):
    fake_chunks = [
        {
            "document": "A",
            "distance": 0.9,
            "product_id": "1",
            "product_name": "Tai nghe Sony",
            "category": "Tai nghe",
            "price": 1000000.0,
            "original_price": 1200000.0,
            "url": "u1",
            "thumbnail_url": "t1",
            "rating": 5.0,
            "review_count": 10,
        },
        {
            "document": "B",
            "distance": 0.3,
            "product_id": "2",
            "product_name": "Chuot gaming",
            "category": "Phu kien",
            "price": 500000.0,
            "original_price": 700000.0,
            "url": "u2",
            "thumbnail_url": "t2",
            "rating": 4.0,
            "review_count": 2,
        },
    ]

    class _FakeEmbedding:
        def tolist(self):
            return [0.1, 0.2]

    class _FakeModel:
        def encode(self, *_args, **_kwargs):
            return _FakeEmbedding()

    monkeypatch.setattr(retriever, "VECTOR_DB", "qdrant")
    monkeypatch.setattr(retriever, "_get_model", lambda: _FakeModel())
    monkeypatch.setattr(retriever, "_qdrant_search", lambda *_args, **_kwargs: fake_chunks)
    monkeypatch.setattr(retriever, "_cohere_rerank", lambda _q, chunks, top_n: chunks)

    result = retriever.retrieve("tai nghe sony", top_k=2)
    assert result[0]["product_id"] == "1"


def test_answer_off_topic_returns_no_sources(monkeypatch):
    monkeypatch.setattr(pipeline, "count_chunks", lambda: 100)
    monkeypatch.setattr(pipeline, "generate", lambda *_args, **_kwargs: "ok")

    result = pipeline.answer("xin chao")
    assert result["sources"] == []
    assert result["chunks"] == []


def test_answer_domain_unsupported_returns_fallback(monkeypatch):
    fake_chunks = [
        {
            "document": "mouse",
            "distance": 0.7,
            "product_name": "Chuot gaming",
            "category": "Phu kien",
            "price": 500000,
            "url": "u",
            "thumbnail_url": "",
            "rating": 0,
            "review_count": 0,
        }
    ]
    monkeypatch.setattr(pipeline, "count_chunks", lambda: 100)
    monkeypatch.setattr(pipeline, "retrieve", lambda *_args, **_kwargs: fake_chunks)

    result = pipeline.answer("laptop gaming rtx 4050")
    assert result["sources"] == []
    assert "laptop gaming" in result["answer"].lower()
