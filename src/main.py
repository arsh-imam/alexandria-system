import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor
from model_engine import ModelEngine
from chat_ui import ChatWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    p = QPalette()
    p.setColor(QPalette.ColorRole.Window,      QColor("#0f0f0f"))
    p.setColor(QPalette.ColorRole.WindowText,  QColor("#e8e8e8"))
    p.setColor(QPalette.ColorRole.Base,        QColor("#1c1c1c"))
    p.setColor(QPalette.ColorRole.Text,        QColor("#e8e8e8"))
    p.setColor(QPalette.ColorRole.Button,      QColor("#1c1c1c"))
    p.setColor(QPalette.ColorRole.ButtonText,  QColor("#e8e8e8"))
    app.setPalette(p)

    engine = ModelEngine()
    # ALEXANDRIA final system runs with source-epistemology fusion ON
    import retriever
    retriever.RETRIEVAL_CONFIG['use_epistemic_fusion'] = True
    window = ChatWindow(engine)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
