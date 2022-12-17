#include <Servo.h>

Servo Left;
Servo Right;
int x;
void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  Serial.setTimeout(1);

  Left.attach(11);
  Right.attach(9);

  pinMode(2, OUTPUT);
  digitalWrite(2,HIGH);
}

void loop() {
  // put your main code here, to run repeatedly:
  digitalWrite(2,HIGH);
  while(!Serial.available()){
    x = Serial.readString().toInt();
  }
  if (x==11){
      Left.write(0);
      Right.write(0);
    }
    else if(x==01){
      Left.write(0);
      Right.write(180);
    }
    else if(x==10){
      Left.write(180);
      Right.write(0);
    }
    else {
      Left.write(0);
      Right.write(0);
    }
  
}
