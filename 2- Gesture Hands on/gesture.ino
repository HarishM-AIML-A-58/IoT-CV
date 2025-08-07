void setup() {
  Serial.begin(9600);

  // Set all pins as OUTPUT
  for (int pin = 2; pin <= 8; pin++) {
    pinMode(pin, OUTPUT);
    digitalWrite(pin, LOW);  // Ensure all are OFF initially
  }
}

void loop() {
  if (Serial.available()) {
    char command = Serial.read();

    // Turn OFF all pins first to prevent overlap
    for (int pin = 2; pin <= 8; pin++) {
      digitalWrite(pin, LOW);
    }

    // Turn ON the corresponding pin
    switch (command) {
      case '1': digitalWrite(2, HIGH); break; // Palm
      case '2': digitalWrite(3, HIGH); break; // Fist
      case '3': digitalWrite(4, HIGH); break; // Two Fingers
      case '4': digitalWrite(5, HIGH); break; // Thumbs Up
      case '5': digitalWrite(6, HIGH); break; // Index Finger
      case '6': digitalWrite(7, HIGH); break; // Rock Sign
      case '7': digitalWrite(8, HIGH); break; // Call Me
      default: break; // Do nothing on unknown input
    }

    delay(500);  // Keep pin ON for a short while
  }
}
