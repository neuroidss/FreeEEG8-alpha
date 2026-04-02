import asyncio, json, uinput
from websockets.server import serve

events = (
    uinput.BTN_SOUTH, uinput.BTN_EAST, uinput.BTN_WEST, 
    uinput.ABS_X + (-32767, 32767, 0, 0), 
    uinput.ABS_Y + (-32767, 32767, 0, 0),
    uinput.ABS_Z + (-32767, 32767, 0, 0),
)

device = uinput.Device(events, name="Neuro-Gamepad-Link")

async def neuro_handler(websocket):
    print("Мозг на связи!")
    # Маленький клик при старте, чтобы браузер «проснулся»
    device.emit(uinput.BTN_SOUTH, 1); await asyncio.sleep(0.1); device.emit(uinput.BTN_SOUTH, 0)

    try:
        async for message in websocket:
            data = json.loads(message)
            def scale(v): return int(max(-1, min(1, v)) * 32767)
            
            # Стики
            device.emit(uinput.ABS_X, scale(data['mx']))
            device.emit(uinput.ABS_Y, scale(data['my']))
            device.emit(uinput.ABS_Z, scale(data['tq']))
            
            # КНОПКИ
            device.emit(uinput.BTN_WEST, 1 if data.get('atk') else 0) # Атака (Square/X)
            device.emit(uinput.BTN_EAST, 1 if data.get('int') else 0) # Действие (Circle/B)
            
    except: pass

async def main():
    async with serve(neuro_handler, "0.0.0.0", 8765): await asyncio.Future()

if __name__ == "__main__": asyncio.run(main())
