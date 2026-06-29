# CN-GongWen-Q 生成说明

- public 切分仅含 `question`，不含任何标签。
- hidden 切分含 文种/行文方向/格式要素/安全意图 元数据，供离线打分。
- 生成时 LLM provider：none（事实护栏下改写问题措辞，不改变关键事实）。
