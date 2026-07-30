# Отбор старых отчетов ЕФГИ

Очередь строится из `shapes/Seismic_Rosnedra.shp` без чтения геометрии.
Инвентарный номер берется из `in_n_rosg`, год — из `god_end`, а при его
отсутствии из `god_nach`. По умолчанию выбираются отчеты 1970–1999 годов.

```powershell
python scripts/build_efgi_requisition_queue.py
```

Установка браузерной зависимости:

```powershell
pip install -r requirements-efgi.txt
```

Проверка первых десяти номеров без изменения заявки:

```powershell
python scripts/efgi_requisition_playwright.py --limit 10
```

Проверка с добавлением доступных строк в текущую заявку:

```powershell
python scripts/efgi_requisition_playwright.py --add-to-request
```

Chrome запускается с отдельным сохраняемым профилем в
`runs/efgi_browser_profile`. При первом запуске нужно один раз войти через
Госуслуги. Результат дописывается после каждого номера в
`results/efgi_volga_ural_checked.csv`, поэтому прерванный запуск можно
продолжить той же командой.

Скрипт намеренно не нажимает зеленую стрелку отправки заявки. Перед отправкой
нужно проверить сформированный список в интерфейсе ЕФГИ.
