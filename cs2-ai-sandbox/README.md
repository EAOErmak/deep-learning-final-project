# CS2 AI Sandbox

Локальный research/playground проект для экспериментов с AI-агентами в Counter-Strike 2 без reverse engineering, memory reading, cheats, VAC bypass и online automation.

Проект специально разделяет:
- `agent` -> решает, какое действие сделать
- `InputController` -> исполняет обычную эмуляцию клавиатуры/мыши
- `StateReader` -> читает состояние игры
- `FeatureEncoder` -> превращает raw state в удобные признаки

Это упрощает дальнейшее подключение:
- imitation learning
- reinforcement learning
- Transformer sequence models

## Ограничения и назначение

- Проект предназначен только для локального sandbox-использования.
- Используйте его на пустой карте, с ботами или в оффлайн-сессии.
- Проект не предназначен для online matchmaking.
- В проекте нет memory reading, reverse engineering или обхода защит.

## Структура проекта

```text
cs2-ai-sandbox/
    main.py
    input_controller.py
    dummy_agent.py
    state_reader.py
    feature_encoder.py
    requirements.txt
    README.md
    demos/
    dataset/
        parsed_demos.json
        raw_ticks/
        events/
    scripts/
        parse_one_demo.py
```

## Установка

Требования:
- Python 3.11+
- Локально установленная Counter-Strike 2

Шаги:

```powershell
cd cs2-ai-sandbox
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Запуск sandbox loop

```powershell
cd cs2-ai-sandbox
python main.py
```

Что делает текущая версия:
- читает mock state из `MockStateReader`
- кодирует его через `encode_state(...)`
- передает features в `DummyAgent`
- получает action dictionary
- `InputController` исполняет действия через обычную эмуляцию клавиатуры/мыши

Loop работает примерно 10 раз в секунду и пишет в консоль:
- текущие features
- текущее action

## Подготовка датасета из demo

Скрипт `scripts/parse_one_demo.py` делает один шаг пайплайна:
- ищет `.dem` в `demos/`
- выбирает первую demo, которой нет в `dataset/parsed_demos.json`
- парсит tick-level данные через `demoparser2`
- парсит базовые events
- сохраняет parquet-файлы в `dataset/raw_ticks/` и `dataset/events/`
- обновляет registry только после успешного сохранения

Запуск:

```powershell
cd cs2-ai-sandbox
python scripts/parse_one_demo.py
```

Результат:
- `dataset/raw_ticks/<demo_name>_ticks.parquet`
- `dataset/events/<demo_name>_<event_name>.parquet`
- `dataset/parsed_demos.json`

Повторный запуск не парсит ту же demo второй раз, если она уже добавлена в registry.

## Как тестировать локально в CS2

Рекомендуемый безопасный сценарий:

1. Запустите CS2 локально.
2. Создайте оффлайн-лобби, тренировочную карту или пустую карту.
3. Убедитесь, что окно игры активно и в фокусе.
4. Запустите `python main.py`.
5. Наблюдайте, как sandbox отправляет обычные keyboard/mouse inputs.
6. Для остановки нажмите `Ctrl+C` в терминале.

Важно:
- сначала протестируйте вне матча
- держите под рукой способ быстро снять фокус с окна игры
- используйте `stop_all()` при штатном завершении, он уже вызывается автоматически

## Архитектура

### `dummy_agent.py`

`DummyAgent` не использует ML. Он циклически делает:
- идти вперед примерно 2 секунды
- повернуть мышь
- идти вправо
- прыжок
- остановка
- выстрел

Агент не нажимает клавиши напрямую. Он возвращает только action dictionary.

### `input_controller.py`

`InputController` исполняет action dictionary и переводит его в:
- удержание movement keys
- разовые действия вроде прыжка
- mouse move
- fire press/release

### `state_reader.py`

Сейчас используется `MockStateReader`, который генерирует fake state:
- позицию игрока
- hp
- money
- ammo
- yaw/pitch
- enemy visible
- enemy relative position

Позже этот слой можно заменить без изменения остальной архитектуры.

### `feature_encoder.py`

`encode_state(raw_state)` возвращает feature dictionary:
- `self_x`
- `self_y`
- `self_z`
- `self_hp`
- `self_money`
- `ammo`
- `yaw`
- `pitch`
- `enemy_visible`
- `enemy_rel_x`
- `enemy_rel_y`
- `enemy_rel_z`
- `enemy_hp`

Если `enemy_visible == False`, то:
- `enemy_rel_x = 0`
- `enemy_rel_y = 0`
- `enemy_rel_z = 0`
- `enemy_hp = 0`

## Как позже подключить реальные источники состояния

### Вариант 1: CS2 GSI

Можно заменить `MockStateReader` на reader, который:
- читает JSON из Game State Integration
- нормализует поля под текущий `raw_state` contract
- оставляет `encode_state(...)` и `DummyAgent` без изменений

Пример направления:
- `GSIStateReader.read_state() -> dict[str, Any]`

### Вариант 2: demoparser2 replay state

Можно сделать reader, который:
- парсит replay/demo
- достает позиции, углы, hp, ammo, видимость или приближенные признаки
- возвращает тот же формат raw state

Это удобно для offline dataset generation и imitation learning pipeline.

## Как позже подключить нейросеть

Вместо `DummyAgent` можно сделать, например:
- `BehaviorCloningAgent`
- `RLAgent`
- `TransformerAgent`

Достаточно сохранить тот же интерфейс:

```python
action = agent.predict(features)
```

То есть новая модель должна:
- принимать features
- возвращать action dictionary

Благодаря этому `main.py` и `InputController` можно почти не менять.

## Идеи для следующего шага

- добавить action smoothing
- добавить configurable keybinds
- добавить emergency hotkey stop
- добавить запись dataset: `state -> action`
- подключить GSI вместо mock state
- вынести action schema в отдельный модуль

## Замечание по безопасности

Этот проект intentionally ограничен обычной пользовательской автоматизацией ввода в локальной среде. Он не предназначен для вмешательства в процесс игры, чтения памяти, обхода античита или использования в online matchmaking.


## Live GSI sandbox setup

???? ?????? ????? ?????????? ? ?????????? CS2 sandbox ????? ??????????? Game State Integration JSON endpoint.

????:
1. ????????? ????????? CS2 dedicated server.
2. ????????? CS2 client ? `-insecure -console -windowed -noborder`.
3. ???????? [config/gamestate_integration_cs2_ai_sandbox.cfg](./config/gamestate_integration_cs2_ai_sandbox.cfg) ? `game/csgo/cfg/`.
4. ????????? Python runtime:

```powershell
python main.py --state-source gsi --gsi-port 3000 --hz 10
```

5. ???????????? ? ?????????? ???????:

```text
connect 127.0.0.1:27015
```

6. ???????? ???? GSI payload / features / action.

?????????:
- GSI ????? ?? ???????? `allplayers` ???????? ??????.
- `allplayers` ???? ???????? ? spectator / observer / GOTV scenario.
- ??????? `visibility_filter` ?????????? approximate visibility: ?????? FOV + distance.
- Raycast / line-of-sight ?? ????? ???????? ????????? ??????.
- ???? ???? ???????????? ?????? ??? ?????????? sandbox / private server ????????.

### ?????? mock ? gsi modes

```powershell
python main.py --state-source mock
python main.py --state-source gsi --gsi-port 3000 --hz 10
```

## Live runtime contract

For local GSI sandbox runtime the Python side now forwards all live fields that are realistically available from the official GSI payload into internal `GameState`.

Current live state includes when available:
- controlled player: position, forward, velocity estimate, hp, armor, money, weapon, ammo, ammo reserve, helmet, flashed/smoked/burning, activity
- map state: map name, mode, phase, round number, CT/T score, losing streaks
- round state: phase, bomb state, phase countdown
- players: `allplayers` when the observer/GOTV scenario provides them
- capability flags: whether position, forward, allplayers, enemy context and spatial state are actually present

Important limitation:
- regular player GSI may still omit `player.position`, `player.forward` and `allplayers`
- in that case the runtime remains functional, but AI will see capability flags that spatial/enemy context is unavailable
- for richer live sandbox context prefer observer / spectator / GOTV-style setups

## Input safety guard

By default, live input is now sent only when the current foreground window title looks like CS2.

Default keywords:
- `counter-strike`
- `cs2`

Practical effect:
- if you `Alt+Tab` away from CS2 to the terminal or another app, the sandbox releases held keys and stops sending new input
- this makes `Ctrl+C` and manual interruption much safer

Examples:

```powershell
python main.py --state-source gsi --agent-mode neural-random
python main.py --state-source gsi --agent-mode neural-random --window-keyword "Counter-Strike 2"
```

If you really want the old behavior, you can disable the guard explicitly:

```powershell
python main.py --state-source gsi --disable-window-guard
```

## Live runtime readiness

The live sandbox runtime now supports explicit readiness levels for GSI-based play:
- `basic`: controlled player exists and activity is `playing`
- `spatial`: additionally requires `player.position` and `player.forward`
- `observer`: additionally requires `allplayers`

Examples:

```powershell
python main.py --state-source gsi --agent-mode pipeline --min-live-readiness basic
python main.py --state-source gsi --agent-mode neural-random --min-live-readiness spatial
python main.py --state-source gsi --agent-mode neural-checkpoint --checkpoint checkpoints\model.pt --min-live-readiness observer
```

One-shot readiness diagnostic:

```powershell
python scripts/check_live_runtime_ready.py
```

If the current GSI payload is not ready enough, the runtime now releases held inputs and waits instead of sending blind actions.
