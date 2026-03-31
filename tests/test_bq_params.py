from google.cloud import bigquery

from arena.data.bigquery.session import BigQuerySession


def test_bq_params_supports_typed_null_float_tuple() -> None:
    session = BigQuerySession.__new__(BigQuerySession)

    params = session._params(
        {
            "tenant_id": "local",
            "outcome_score": ("FLOAT64", None),
            "importance_score": ("FLOAT64", 0.75),
        }
    )

    by_name = {param.name: param for param in params}

    assert isinstance(by_name["tenant_id"], bigquery.ScalarQueryParameter)
    assert by_name["tenant_id"].type_ == "STRING"
    assert by_name["outcome_score"].type_ == "FLOAT64"
    assert by_name["outcome_score"].value is None
    assert by_name["importance_score"].type_ == "FLOAT64"
    assert by_name["importance_score"].value == 0.75
