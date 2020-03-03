#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>

#include "data.h"

int main(int argc, char** argv)
{
  QGuiApplication app(argc, argv);

  QQmlApplicationEngine engine;
  Data data;
  engine.rootContext()->setContextProperty("hoge", &data);
  engine.load(QUrl(QStringLiteral("qrc:/main.qml")));

  return app.exec();
}