from pathlib import Path

from app.config.loader import load_bank_registry


def test_load_bank_registry_reads_selected_banks() -> None:
    config_path = Path("app/config/banks.yaml")

    registry = load_bank_registry(config_path)

    assert registry.version == 1
    assert len(registry.banks) == 3
    assert registry.get_bank("unibank") is not None
    assert registry.get_bank("aeb") is not None
    assert registry.get_bank("vtb") is not None
    assert set(registry.get_bank("unibank").topics.keys()) == {
        "credits",
        "deposits",
        "branches",
    }
    assert registry.get_bank("unibank").topics["credits"].follow_links is False
    assert len(registry.get_bank("unibank").topics["credits"].document_urls) == 19
