from datetime import datetime

from site_inspector.parser import HTMLParser


def parse(html):
    return HTMLParser("https://example.com").parse(
        url="https://example.com/pages/home",
        html=html,
        depth=0,
        status_code=200,
        content_type="text/html",
        fetched_at=datetime(2026, 10, 17),
    )[0]


def test_non_http_images_are_retained_without_download_url():
    page = parse('''
        <img src="data:image/png;base64,aGVsbG8=" alt="embedded" width="10">
        <img src="blob:https://example.com/image">
        <img src="http://">
        <img src="../photo.png" alt="photo">
    ''')
    assert len(page.images) == 4
    assert page.images[0].src == "data:image/png;base64,aGVsbG8="
    assert page.images[0].alt_text == "embedded"
    assert page.images[0].width == 10
    assert all(image.absolute_url is None for image in page.images[:3])
    assert str(page.images[3].absolute_url) == "https://example.com/photo.png"


def test_word_count_uses_full_text_while_preview_stays_bounded():
    page = parse("<p>" + "word " * 1000 + "</p>")
    assert page.word_count == 1000
    assert len(page.text_preview) == 1200
    assert page.text_preview.endswith("...")
