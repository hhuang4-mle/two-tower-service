CREATE TEMP TABLE mapped AS
SELECT source AS source_record, {embedding_fields},
       ARRAY(SELECT field FROM UNNEST([{failed_fields}]) AS field
             WHERE field IS NOT NULL) AS failed_fields
FROM `{source_table}` AS source
{embedding_joins};

ASSERT NOT EXISTS (
  SELECT 1 FROM `{source_table}` AS source
  WHERE {unstaged_texts}
) AS 'Source contains unstaged texts; source must remain unchanged during the run';

CREATE TEMP TABLE successes AS
SELECT {source_fields}, {embedding_columns}
FROM mapped WHERE ARRAY_LENGTH(failed_fields) = 0;

CREATE TEMP TABLE failures AS
SELECT TO_JSON(source_record) AS source_record, failed_fields,
       'text_embedding' AS stage, 'UNRESOLVED_PREDICTION' AS error_code,
       {attempts} AS attempts,
       'Missing, duplicate, or invalid text prediction after retries' AS message
FROM mapped WHERE ARRAY_LENGTH(failed_fields) > 0;

ASSERT (
  (SELECT COUNT(*) FROM successes) + (SELECT COUNT(*) FROM failures)
) = (SELECT COUNT(*) FROM `{source_table}`)
AS 'Publication must account for every source row';

CREATE TABLE IF NOT EXISTS `{output_table}` AS SELECT * FROM successes WHERE FALSE;
CREATE TABLE IF NOT EXISTS `{failure_table}` AS SELECT * FROM failures WHERE FALSE;

BEGIN TRANSACTION;
TRUNCATE TABLE `{output_table}`;
INSERT INTO `{output_table}` ({output_columns}) SELECT * FROM successes;
TRUNCATE TABLE `{failure_table}`;
INSERT INTO `{failure_table}`
  (source_record, failed_fields, stage, error_code, attempts, message)
SELECT * FROM failures;
COMMIT TRANSACTION;
