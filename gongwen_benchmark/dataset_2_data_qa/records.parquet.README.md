# records.parquet（本地生成，不入库）

PR 评审系统不支持二进制 diff，因此仓库仅提交可审阅的 records.csv。
如需 Parquet，请本地运行：

```bash
python gongwen_benchmark/scripts/generate_benchmarks.py \
  --profile standard --export-parquet /tmp/gongwen-records.parquet
```
