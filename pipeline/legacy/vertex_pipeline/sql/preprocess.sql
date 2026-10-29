CREATE OR REPLACE TABLE `{input_table}` AS
SELECT TO_JSON_STRING(STRUCT(text AS text)) AS instances
FROM (
  SELECT DISTINCT
    COLLATE(COALESCE(CAST(source.`{text_column}` AS STRING), ''), '') AS text
  FROM `{source_table}` AS source
)
