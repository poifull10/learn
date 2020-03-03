#pragma once

#include <QObject>
#include <QString>

class Data : public QObject
{
  Q_OBJECT
public:
  Q_INVOKABLE QString getTextFromCpp() { return QString("TEST DAYO"); }
};
