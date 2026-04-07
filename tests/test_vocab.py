from whisper_guard.vocab import build_hotwords_prompt, filter_filler_words


def test_build_hotwords_prompt():
    result = build_hotwords_prompt(["Furutech", "NCF", "Alpha"])
    assert result == "Furutech NCF Alpha"


def test_build_hotwords_prompt_strips_whitespace():
    result = build_hotwords_prompt(["  hello ", "", " world "])
    assert result == "hello world"


def test_build_hotwords_prompt_empty():
    result = build_hotwords_prompt([])
    assert result == ""


def test_filter_filler_words_defaults():
    text = "這是嗯嗯一段啊啊測試呃呃文字喔喔結束"
    result = filter_filler_words(text)
    assert "嗯嗯" not in result
    assert "啊啊" not in result
    assert "呃呃" not in result
    assert "喔喔" not in result
    assert "測試" in result


def test_filter_filler_words_custom():
    result = filter_filler_words("那個就是這樣", ["那個", "就是"])
    assert result == "這樣"


def test_filter_filler_words_no_match():
    result = filter_filler_words("正常文字沒有贅詞")
    assert result == "正常文字沒有贅詞"
