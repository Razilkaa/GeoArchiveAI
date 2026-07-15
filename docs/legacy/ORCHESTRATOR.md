# Report operator

`report_orchestrator.py` управляет специализированными агентами поверх существующего `job.json`.

```text
classifier
   ├── rag ─────┐
   └── map ─────┼── qc ── export
                ┘
```

- `rag` и `map` запускаются независимо.
- `qc` требует RAG и наблюдает картографа: если карта заблокирована, выпускается частичный QC.
- После появления карты повторный запуск автоматически пересчитывает QC и экспорт.
- Каждый агент сохраняет результат в общей оболочке `status / artifacts / metrics / issues`.
- Состояние атомарно записывается в `runs/<report_id>/orchestrator/state.json`.
- Заблокированный агент можно безопасно повторить; завершённые агенты с существующими артефактами не запускаются снова.

## Команды

```powershell
python pipeline/report_orchestrator.py runs/384092/job.json plan
python pipeline/report_orchestrator.py runs/384092/job.json run
python pipeline/report_orchestrator.py runs/384092/job.json status
python pipeline/report_orchestrator.py runs/384092/job.json run --force qc
```

## Контракт картографа

Картографический адаптер пишет `runs/<report_id>/map_agent/result.json`. Пример находится в `pipeline/contracts/map_agent_result.example.json`. Артефакты должны существовать на диске; иначе оператор оставляет агент заблокированным.

Оператор не выполняет внешние запросы самостоятельно. Сетевые исполнители обязаны проверять `runs/<report_id>/privacy.json` до запуска.
