#include <Servo.h>

Servo Left;
Servo Right;
int x;
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  Serial.setTimeout(1);

  Left.attach(11);
  Right.attach(9);
}

void loop() {
  // put your main code here, to run repeatedly:
  while(Serial.available()==0){
  }
  x = Serial.parseInt();
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
    delay(10);
  
}
