from site_inspector.config import CrawlConfig


def test_allowed_domains_defaults_to_base_host():
    config = CrawlConfig(base_url="https://example.com")
    assert config.allowed_domains == ["example.com"]
    assert config.normalized_domains() == {"example.com"}
