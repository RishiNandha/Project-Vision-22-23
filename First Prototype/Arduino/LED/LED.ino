void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  Serial.setTimeout(1);
  pinMode(2, OUTPUT);
  pinMode(3, OUTPUT); 
}

void loop() {
  // put your main code here, to run repeatedly:
  while(!Serial.available()){
    int x = Serial.readString().toInt();
    if (x==11){
      
  digitalWrite(2, HIGH);
  digitalWrite(3, HIGH);
    }
    else if(x==01){
    
  digitalWrite(2, LOW);
  digitalWrite(3, HIGH);  
    }
    else if(x==10){
     
  digitalWrite(2, HIGH);
  digitalWrite(3, LOW); 
    }
    else {
      
  digitalWrite(2, LOW);
  digitalWrite(3, LOW);
    }
  }
  
}
