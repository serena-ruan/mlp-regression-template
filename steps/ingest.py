"""
This module defines the following routines used by the 'ingest' step of the regression pipeline:

- ``load_file_as_dataframe``: Defines customizable logic for parsing dataset formats that are not
  natively parsed by MLflow Pipelines (i.e. formats other than Parquet, Delta, and Spark SQL).
"""

import logging

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession

_logger = logging.getLogger(__name__)


def load_file_as_dataframe(file_path: str, file_format: str) -> DataFrame:
    """
    Load content from the specified dataset file as a Pandas DataFrame.

    This method is used to load dataset types that are not natively  managed by MLflow Pipelines
    (datasets that are not in Parquet, Delta Table, or Spark SQL Table format). This method is
    called once for each file in the dataset, and MLflow Pipelines automatically combines the
    resulting DataFrames together.

    :param file_path: The path to the dataset file.
    :param file_format: The file format string, such as "csv".
    :return: A Pandas DataFrame representing the content of the specified file.
    """

    if file_format == "csv":
        spark = SparkSession.builder.getOrCreate()
        return spark.read.format("csv").option("header", True).option("inferSchema", True).load(file_path)
    else:
        raise NotImplementedError
