import json
import math

import pytest
from poi_batch_panel_builder import main as batch_main
from poi_batch_panel_builder import parse_args as parse_batch_args
from poi_batch_panel_builder import resolve_cities, validate_years
from poi_panel_builder import (
    category_from_gdb_path,
    classify_poi_source,
    fix_zip_name,
    is_chain_brand,
    map_poi_category,
    normalize_crs_to_wgs84,
    normalize_csv_poi_chunk,
    parse_args,
    shannon_entropy,
)
from poi_panel_builder import validate_years as validate_csv_years

from urban_intervention.pipelines.poi.batch import CityBatch, build_batch_year, make_city_batches
from urban_intervention.pipelines.poi.cache import _gdb_mtime
from urban_intervention.pipelines.poi.gdb import (
    ExtractedGdbSource,
    category_from_2020_fields,
    discover_extracted_gdb_sources,
    is_nested_gdb,
    normalize_gdb_category,
    sources_from_2020_layers,
)
from urban_intervention.pipelines.poi.normalize import category_series_from_gdb
from urban_intervention.pipelines.poi.sources import city_name_variants


def test_fix_zip_name_repairs_double_mojibake():
    assert fix_zip_name("楂樺痉poi鏁版嵁/椁愰ギ鏈嶅姟.gdb.zip") == ("高德poi数据/餐饮服务.gdb.zip")


@pytest.mark.parametrize(
    "cate_a,expected",
    [
        ("餐饮服务", "food"),
        ("购物服务", "retail"),
        ("生活服务", "life_service"),
        ("体育休闲服务", "leisure"),
        ("公司企业", "office_enterprise"),
        ("交通设施服务", "transport"),
        ("未知分类", "other"),
    ],
)
def test_map_poi_category(cate_a, expected):
    assert map_poi_category(cate_a) == expected


def test_shannon_entropy_ignores_zero_counts():
    assert shannon_entropy([10, 0, 0]) == pytest.approx(0.0)
    assert shannon_entropy([1, 1, 1, 1]) == pytest.approx(math.log(4))


def test_is_chain_brand_matches_common_brand_names():
    assert is_chain_brand("星巴克咖啡 北京站店")
    assert is_chain_brand("瑞幸咖啡(国贸店)")
    assert not is_chain_brand("东城区社区便利店")


def test_classify_extracted_poi_sources():
    assert (
        classify_poi_source("data/高德poi数据/2012-2017-WGS84/2012-84/北京市_wgs84.csv")
        == "city_csv"
    )
    assert classify_poi_source("data/高德poi数据/2018/购物服务.gdb") == "filegdb_dir"
    assert classify_poi_source("data/高德poi数据/2021/购物服务.gdb.zip") == "filegdb_zip"
    assert classify_poi_source("data/高德poi数据/2024/全国购物服务2024.gdb.rar") == "filegdb_rar"


def test_city_name_variants_adds_city_suffix_for_zhou_cities():
    assert "常州市" in city_name_variants("changzhou")


def test_category_from_gdb_path():
    assert category_from_gdb_path("data/高德poi数据/2018/购物服务.gdb") == "购物服务"
    assert category_from_gdb_path("data/高德poi数据/2024/全国餐饮服务2024.gdb.rar") == "餐饮服务"


def test_is_nested_gdb_detects_parent_gdb():
    from pathlib import Path

    assert is_nested_gdb(Path("2019/全国餐饮服务.gdb/全国2019餐饮服务.gdb"))
    assert not is_nested_gdb(Path("2019/全国购物服务.gdb"))


def test_discover_extracted_gdb_sources_prefers_nested_valid_gdb(tmp_path):
    outer = tmp_path / "2019" / "全国餐饮服务.gdb"
    inner = outer / "全国2019餐饮服务.gdb"
    inner.mkdir(parents=True)
    (inner / "a00000001.gdbtable").write_bytes(b"x")
    (inner / "a00000001.gdbtablx").write_bytes(b"x")
    normal = tmp_path / "2019" / "全国购物服务.gdb"
    normal.mkdir(parents=True)
    (normal / "a00000001.gdbtable").write_bytes(b"x")
    (normal / "a00000001.gdbtablx").write_bytes(b"x")

    sources = discover_extracted_gdb_sources(base_dir=tmp_path, year=2019)
    rel_paths = {source.path.relative_to(tmp_path).as_posix() for source in sources}

    assert rel_paths == {
        "2019/全国餐饮服务.gdb/全国2019餐饮服务.gdb",
        "2019/全国购物服务.gdb",
    }


def test_empty_gdb_mtime_has_traceable_error(tmp_path):
    empty = tmp_path / "empty.gdb"
    empty.mkdir()
    source = ExtractedGdbSource(empty, 2019, None, False)
    with pytest.raises(ValueError, match="contains no files"):
        _gdb_mtime(source)


def test_discovery_skips_gdb_with_unknown_year(tmp_path):
    unknown = tmp_path / "unknown.gdb"
    unknown.mkdir()
    (unknown / "a.gdbtable").write_bytes(b"x")
    (unknown / "a.gdbtablx").write_bytes(b"x")
    with pytest.warns(RuntimeWarning, match="year cannot be inferred"):
        sources = discover_extracted_gdb_sources(base_dir=tmp_path, year=None)
    assert sources == []


def test_category_from_2020_fields_prefers_major_type_prefix():
    import pandas as pd

    row = pd.Series(
        {
            "typename": "餐饮服务;中餐厅;综合酒楼",
            "typecode": "050100",
            "tag": "",
        }
    )
    assert category_from_2020_fields(row) == "餐饮服务"


def test_normalize_gdb_category_fixes_2020_layer_aliases():
    assert normalize_gdb_category("体育休闲") == "体育休闲服务"
    assert normalize_gdb_category("科教文化") == "科教文化服务"
    assert normalize_gdb_category("政府机构与社会团体") == "政府机构及社会团体"


def test_sources_from_2020_layers_expands_each_layer(tmp_path):
    gdb_path = tmp_path / "2020" / "2020.gdb"
    gdb_path.mkdir(parents=True)
    sources = sources_from_2020_layers(gdb_path, ["餐饮服务", "体育休闲"])

    assert [(source.category, source.layer) for source in sources] == [
        ("餐饮服务", "餐饮服务"),
        ("体育休闲服务", "体育休闲"),
    ]


def test_category_series_from_gdb_uses_override_category():
    import pandas as pd

    df = pd.DataFrame({"typename": ["未知"], "typecode": ["000000"]})
    out = category_series_from_gdb(
        df, fallback_path="x/2021/购物服务.gdb", category_override="餐饮服务"
    )
    assert out.tolist() == ["餐饮服务"]


def test_category_series_from_gdb_can_infer_2020_rows():
    import pandas as pd

    df = pd.DataFrame(
        {
            "typename": ["餐饮服务;中餐厅", ""],
            "typecode": ["050100", "060100"],
        }
    )
    out = category_series_from_gdb(df, fallback_path="x/2020/2020.gdb", infer_from_fields=True)
    assert out.tolist() == ["餐饮服务", "购物服务"]


def test_read_filegdb_signature_accepts_layer_argument():
    import inspect

    from urban_intervention.pipelines.poi.normalize import read_filegdb

    assert "layer" in inspect.signature(read_filegdb).parameters


def test_normalize_csv_poi_chunk():
    import pandas as pd

    raw = pd.DataFrame(
        {
            "name": ["星巴克咖啡"],
            "province": ["北京市"],
            "city": ["北京市"],
            "district": ["朝阳区"],
            "code": [110105],
            "lon": [116.46],
            "lat": [39.92],
            "typecode": ["050500"],
            "cate_A": ["餐饮服务"],
            "cate_B": ["咖啡厅"],
            "cate_C": ["星巴克咖啡"],
        }
    )
    out = normalize_csv_poi_chunk(raw, year=2012, city_key="beijing")
    assert list(out.columns) == [
        "city",
        "year",
        "name",
        "lon",
        "lat",
        "typecode",
        "cate_A",
        "cate_B",
        "cate_C",
        "category",
        "is_commercial",
        "is_chain",
        "is_community_commerce",
    ]
    assert out.iloc[0]["category"] == "food"
    assert out.iloc[0]["is_chain"] == 1


def test_normalize_crs_to_wgs84_projects_metric_points():
    import geopandas as gpd
    from shapely.geometry import Point

    gdf = gpd.GeoDataFrame(
        {"name": ["x"]},
        geometry=[Point(12968770.0, 4851421.0)],
        crs="EPSG:3857",
    )
    out, method = normalize_crs_to_wgs84(gdf)
    assert method == "to_crs:EPSG:3857->EPSG:4326"
    assert out.crs.to_epsg() == 4326
    assert 116 <= out.geometry.iloc[0].x <= 117
    assert 39 <= out.geometry.iloc[0].y <= 40


def test_poi_audit_columns():
    from audit_poi_panel import audit_city_panel_columns

    assert audit_city_panel_columns() == [
        "city",
        "year",
        "n_grids_with_poi",
        "poi_count_total",
        "food_total",
        "retail_total",
        "life_service_total",
        "chain_total",
        "median_grid_poi_count",
    ]


def test_poi_panel_builder_accepts_dry_run_flag():
    args = parse_args(["--city", "beijing", "--years", "2017", "--dry-run"])
    assert args.dry_run is True


def test_city_csv_builder_rejects_filegdb_years():
    validate_csv_years([2012, 2017])
    with pytest.raises(ValueError, match="poi_batch_panel_builder.py"):
        validate_csv_years([2017, 2018])


def test_batch_panel_builder_accepts_batch_controls():
    args = parse_batch_args(
        [
            "--city",
            "beijing,shanghai",
            "--years",
            "2020",
            "--batch-size",
            "2",
            "--batch-index",
            "1",
            "--dry-run",
        ]
    )
    assert args.city == "beijing,shanghai"
    assert args.years == "2020"
    assert args.batch_size == 2
    assert args.batch_index == 1
    assert args.dry_run is True


def test_make_city_batches_orders_by_longitude_and_builds_bbox():
    batches = make_city_batches(["shanghai", "beijing", "chengdu"], batch_size=2)

    assert [batch.cities for batch in batches] == [
        ("chengdu", "beijing"),
        ("shanghai",),
    ]
    assert batches[0].bbox == (103.47, 29.97, 117.0, 40.5)


def test_batch_resolve_cities_deduplicates_user_input():
    assert resolve_cities("beijing,shanghai,beijing") == ["beijing", "shanghai"]


def test_batch_builder_rejects_pre_gdb_years():
    with pytest.raises(ValueError, match="2018"):
        validate_years([2017, 2018])


def test_normalize_crs_to_wgs84_allows_empty_missing_crs_frame():
    import geopandas as gpd

    gdf = gpd.GeoDataFrame({"name": []}, geometry=[], crs=None)
    out, method = normalize_crs_to_wgs84(gdf)

    assert out.empty
    assert out.crs.to_epsg() == 4326
    assert method == "assume_wgs84:empty_missing_crs"


def test_build_batch_year_reads_with_loaded_grid_bounds(monkeypatch):
    from types import SimpleNamespace

    import urban_intervention.pipelines.poi.batch as batch_module

    def fake_load_city_grids(_cities):
        return SimpleNamespace(total_bounds=[1.0, 2.0, 3.0, 4.0])

    def fake_discover_gdb_sources(year=None, categories=None, base_dir=None):
        return []

    monkeypatch.setattr(batch_module, "load_city_grids", fake_load_city_grids)
    monkeypatch.setattr(batch_module, "discover_extracted_gdb_sources", fake_discover_gdb_sources)

    from pytest import raises

    with raises(FileNotFoundError):
        build_batch_year(
            CityBatch(index=1, cities=("beijing",), bbox=(115.8, 39.3, 117.0, 40.5)),
            2020,
        )


def test_build_batch_year_fails_closed_when_a_source_fails(monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    import urban_intervention.pipelines.poi.batch as batch_module

    source = ExtractedGdbSource(Path("broken.gdb"), 2020, "餐饮服务", False)
    monkeypatch.setattr(
        batch_module,
        "load_city_grids",
        lambda _cities: SimpleNamespace(total_bounds=[1.0, 2.0, 3.0, 4.0]),
    )
    monkeypatch.setattr(
        batch_module,
        "discover_extracted_gdb_sources",
        lambda year=None, categories=None: [source],
    )

    def fail_source(*_args, **_kwargs):
        raise OSError("unreadable source")

    monkeypatch.setattr(batch_module, "_process_gdb", fail_source)

    with pytest.raises(RuntimeError, match="refusing to finalize a partial panel"):
        build_batch_year(
            CityBatch(index=1, cities=("beijing",), bbox=(115.8, 39.3, 117.0, 40.5)),
            2020,
            workers=1,
            use_cache=False,
        )


def test_process_gdb_forwards_refresh_to_cache(monkeypatch, tmp_path):
    import pandas as pd

    import urban_intervention.pipelines.poi.batch as batch_module
    import urban_intervention.pipelines.poi.cache as cache_module

    source = ExtractedGdbSource(tmp_path / "source.gdb", 2020, "餐饮服务", False)
    observed = {}

    def fake_read_source_cached(*_args, **kwargs):
        observed.update(kwargs)
        return pd.DataFrame(), "source", "cached"

    monkeypatch.setattr(cache_module, "read_source_cached", fake_read_source_cached)
    monkeypatch.setattr(
        batch_module, "aggregate_chunk_multi_city", lambda _frame, _grids: pd.DataFrame()
    )

    batch_module._process_gdb(
        source,
        2020,
        "batch-01",
        (1.0, 2.0, 3.0, 4.0),
        object(),
        use_cache=True,
        refresh_cache=True,
    )

    assert observed["refresh"] is True


def test_batch_cli_forwards_refresh_cache(monkeypatch):
    import pandas as pd
    import poi_batch_panel_builder as batch_script

    observed = {}

    def fake_build_batch_year(*_args, **kwargs):
        observed.update(kwargs)
        return pd.DataFrame(columns=["city"])

    monkeypatch.setattr(batch_script, "build_batch_year", fake_build_batch_year)

    assert (
        batch_main(
            [
                "--city",
                "beijing",
                "--years",
                "2020",
                "--batch-size",
                "1",
                "--refresh-cache",
                "--dry-run",
            ]
        )
        == 0
    )
    assert observed["refresh_cache"] is True


def test_single_and_multi_city_aggregation_are_equivalent_for_one_city():
    import geopandas as gpd
    import pandas as pd
    from pandas.testing import assert_frame_equal
    from shapely.geometry import box

    from urban_intervention.pipelines.poi.aggregate import (
        aggregate_chunk,
        aggregate_chunk_multi_city,
    )

    points = pd.DataFrame(
        {
            "name": ["a", "b"],
            "category": ["food", "retail"],
            "is_commercial": [1, 1],
            "is_chain": [0, 1],
            "is_community_commerce": [1, 0],
            "lon": [0.25, 0.75],
            "lat": [0.25, 0.75],
        }
    )
    single_grid = gpd.GeoDataFrame(
        {"grid_id": ["g1"]}, geometry=[box(0.0, 0.0, 1.0, 1.0)], crs="EPSG:4326"
    )
    multi_grid = single_grid.copy()
    multi_grid.insert(0, "city", "beijing")

    single = aggregate_chunk(points, single_grid).sort_index(axis=1)
    multi = (
        aggregate_chunk_multi_city(points, multi_grid)
        .drop(columns="city")
        .sort_index(axis=1)
    )

    assert_frame_equal(single, multi)


def test_save_city_panel_writes_via_temporary_file():
    import inspect

    from urban_intervention.pipelines.poi.aggregate import save_city_panel

    source = inspect.getsource(save_city_panel)
    assert ".tmp" in source
    assert ".replace(" in source


def test_save_city_panel_preserves_other_years_under_lock(tmp_path, monkeypatch):
    import pandas as pd

    import urban_intervention.pipelines.poi.aggregate as aggregate_module

    monkeypatch.setattr(aggregate_module, "OUT_DIR", tmp_path)
    base = {"city": ["beijing"], "grid_id": ["g1"], "poi_count": [10]}
    aggregate_module.save_city_panel("beijing", [pd.DataFrame({**base, "year": [2019]})])
    aggregate_module.save_city_panel(
        "beijing", [pd.DataFrame({**base, "year": [2020], "poi_count": [12]})]
    )

    result = pd.read_parquet(tmp_path / "beijing_poi_grid_yearly.parquet")
    assert result.year.tolist() == [2019, 2020]


def test_save_city_panel_writes_and_merges_year_provenance(tmp_path, monkeypatch):
    import pandas as pd

    import urban_intervention.pipelines.poi.aggregate as aggregate_module

    monkeypatch.setattr(aggregate_module, "OUT_DIR", tmp_path)
    base = {"city": ["beijing"], "grid_id": ["g1"], "poi_count": [10]}
    frame_2017 = pd.DataFrame({**base, "year": [2017]})
    frame_2017.attrs["poi_provenance"] = {
        "2017": {
            "producer": "poi_panel_builder",
            "source_format": "city_csv",
            "sources": ["2017/beijing.csv"],
        }
    }
    frame_2018 = pd.DataFrame({**base, "year": [2018]})
    frame_2018.attrs["poi_provenance"] = {
        "2018": {
            "producer": "poi_batch_panel_builder",
            "source_format": "filegdb",
            "sources": [{"path": "2018/food.gdb", "category": "餐饮服务", "layer": None}],
        }
    }

    aggregate_module.save_city_panel("beijing", [frame_2017])
    aggregate_module.save_city_panel("beijing", [frame_2018])

    provenance = json.loads(
        (tmp_path / "beijing_poi_grid_yearly.provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["city"] == "beijing"
    assert provenance["years"]["2017"]["producer"] == "poi_panel_builder"
    assert provenance["years"]["2018"]["producer"] == "poi_batch_panel_builder"
