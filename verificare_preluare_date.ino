#include <Arduino.h>
#include <bme68xLibrary.h>
#include <commMux.h>
#include <SPI.h>
#include <SD.h>
#include "mlp_model.h"  // Include header generat din MATLAB

#define N_KIT_SENS 8
#define SD_PIN_CS 33
#define PANIC_LED LED_BUILTIN
#define PANIC_DUR 1000
#define MEAS_DUR 140
#define LOG_FILE_NAME "/data.csv"
#define N_STEPS 10

bool isTrainingMode = false;  // ⚠️ Setează true pentru colectare, false pentru inferență

Bme68x bme[N_KIT_SENS];
commMux commSetup[N_KIT_SENS];
uint8_t lastMeasindex[N_KIT_SENS] = {0};
bme68xData sensorData[N_KIT_SENS] = {0};
bme68xData sensorBuffer[N_KIT_SENS][N_STEPS];
bool stepReceived[N_KIT_SENS][N_STEPS] = {false};
bool cycleComplete[N_KIT_SENS] = {false};

String logHeader;
uint32_t lastLogged = 0;

File file;

void panicLeds();
void appendFile(const String &sensorData);

void setup(void) {
  Serial.begin(115200);
  commMuxBegin(Wire, SPI);
  pinMode(PANIC_LED, OUTPUT);
  delay(100);

  if (!SD.begin(SD_PIN_CS)) {
    Serial.println("SD Card not found");
    panicLeds();
  } else {
    if (SD.exists(LOG_FILE_NAME)) {
      SD.remove(LOG_FILE_NAME);
    }
    file = SD.open(LOG_FILE_NAME, FILE_WRITE);
    if (!file) {
      Serial.println("Failed to open file for writing");
      panicLeds();
    }
    logHeader = "TimeStamp(ms),Sensor Index";
    for (int step = 0; step < N_STEPS; step++) {
      logHeader += ",GasRes_" + String(step);
      logHeader += ",GasIdx_" + String(step);
      logHeader += ",MeasIdx_" + String(step);
    }
    logHeader += "\n";

    if (file.print(logHeader)) {
      Serial.print(logHeader);
      file.close();
    } else {
      panicLeds();
    }
    logHeader = "";
  }

  for (uint8_t i = 0; i < N_KIT_SENS; i++) {
    commSetup[i] = commMuxSetConfig(Wire, SPI, i, commSetup[i]);
    bme[i].begin(BME68X_SPI_INTF, commMuxRead, commMuxWrite, commMuxDelay, &commSetup[i]);
    if (bme[i].checkStatus()) {
      Serial.println("Initializing sensor " + String(i) + " failed with error " + bme[i].statusString());
      panicLeds();
    }
    bme[i].setTPH();

    uint16_t tempProf[10] = {320, 100, 100, 100, 200, 200, 200, 320, 320, 320};
    uint16_t mulProf[10] = {5, 2, 10, 30, 5, 5, 5, 5, 5, 5};
    uint16_t sharedHeatrDur = MEAS_DUR - (bme[i].getMeasDur(BME68X_PARALLEL_MODE) / 1000);
    bme[i].setHeaterProf(tempProf, mulProf, sharedHeatrDur, 10);
    bme[i].setOpMode(BME68X_PARALLEL_MODE);

    for (int s = 0; s < N_STEPS; s++) {
      stepReceived[i][s] = false;
    }
    cycleComplete[i] = false;
  }
}

void loop(void) {
  if ((millis() - lastLogged) < MEAS_DUR) return;
  lastLogged = millis();

  for (uint8_t i = 0; i < N_KIT_SENS; i++) {
    if (bme[i].fetchData()) {
      uint8_t nFieldsLeft;
      do {
        nFieldsLeft = bme[i].getData(sensorData[i]);
        if (sensorData[i].status & BME68X_NEW_DATA_MSK) {
          uint8_t step = sensorData[i].gas_index;
          sensorBuffer[i][step] = sensorData[i];
          stepReceived[i][step] = true;

          bool allStepsReceived = true;
          for (int s = 0; s < N_STEPS; s++) {
            if (!stepReceived[i][s]) {
              allStepsReceived = false;
              break;
            }
          }
          if (allStepsReceived && !cycleComplete[i]) {
            cycleComplete[i] = true;
            Serial.printf("Senzor %d a terminat un ciclu complet.\n", i);
          }
        }
      } while (nFieldsLeft);
    }

    if (cycleComplete[i]) {
      if (isTrainingMode) {
        String line = String(millis()) + "," + String(i);
        for (int s = 0; s < N_STEPS; s++) {
          bme68xData &d = sensorBuffer[i][s];
          line += "," + String(d.gas_resistance, 0);
          line += "," + String(d.gas_index);
          line += "," + String(d.meas_index);
        }
        appendFile(line + "\n");
      } else {
        float avg_temp = 0, avg_hum = 0, avg_press = 0;
        float features[N_IN];

        for (int s = 0; s < N_STEPS; s++) {
          avg_temp += sensorBuffer[i][s].temperature;
          avg_hum  += sensorBuffer[i][s].humidity;
          avg_press += sensorBuffer[i][s].pressure;
        }
        avg_temp /= N_STEPS;
        avg_hum  /= N_STEPS;
        avg_press /= N_STEPS;

        features[0] = (avg_temp - mu[0]) / sigma[0];
        features[1] = (avg_hum - mu[1]) / sigma[1];
        features[2] = (avg_press - mu[2]) / sigma[2];

        int k = 3;
        for (int s = 0; s < N_STEPS; s++) {
          float gas = sensorBuffer[i][s].gas_resistance;
          if (gas < 0) gas = 0;
          features[k] = (gas - mu[k]) / sigma[k];
          k++;
        }

        float probs[N_OUT];
        int predicted;
        mlp_predict_vector(features, probs, predicted);

        Serial.printf(">> Senzor %d: Clasă detectată = %d (%.2f%%)\n", i, predicted + 1, probs[predicted] * 100);
      }

      for (int s = 0; s < N_STEPS; s++) stepReceived[i][s] = false;
      cycleComplete[i] = false;
    }
  }
}

void panicLeds() {
  while (1) {
    digitalWrite(PANIC_LED, HIGH);
    delay(PANIC_DUR);
    digitalWrite(PANIC_LED, LOW);
    delay(PANIC_DUR);
  }
}

void appendFile(const String &sensorData) {
  file = SD.open(LOG_FILE_NAME, FILE_WRITE);
  if (!file) {
    Serial.println("Failed to open file for appending");
    panicLeds();
  }
  if (file.print(sensorData)) {
    Serial.print(sensorData);
  } else {
    Serial.println("Write append failed");
  }
  file.close();
}
