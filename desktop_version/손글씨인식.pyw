"""윈도우에서 더블클릭하면 검은 명령 창 없이 손글씨 인식 앱만 띄운다 (.pyw 는 pythonw로 실행됨)."""

import os
import runpy

폴더 = os.path.dirname(os.path.abspath(__file__))
os.chdir(폴더)
runpy.run_path(os.path.join(폴더, "app.py"), run_name="__main__")
