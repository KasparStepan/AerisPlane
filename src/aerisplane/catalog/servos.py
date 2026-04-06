"""Servo catalog entries."""
from aerisplane.core.control_surface import Servo

hitec_hs65mg = Servo("Hitec HS-65MG", torque=0.196, speed=500.0, voltage=6.0, mass=0.013)
hitec_hs5086wb = Servo("Hitec HS-5086WB", torque=0.275, speed=500.0, voltage=6.0, mass=0.019)
hitec_hs7950th = Servo("Hitec HS-7950TH", torque=0.686, speed=500.0, voltage=6.0, mass=0.068)
savox_sh0255mg = Servo("Savox SH-0255MG", torque=0.167, speed=500.0, voltage=4.8, mass=0.014)
savox_sc1256tg = Servo("Savox SC-1256TG", torque=0.686, speed=600.0, voltage=6.0, mass=0.056)
futaba_s3003 = Servo("Futaba S3003", torque=0.318, speed=400.0, voltage=4.8, mass=0.037)
futaba_s3305 = Servo("Futaba S3305", torque=0.490, speed=500.0, voltage=6.0, mass=0.042)
towerpro_mg996r = Servo("TowerPro MG996R", torque=0.980, speed=333.0, voltage=6.0, mass=0.055)
kst_x08h = Servo("KST X08H V5", torque=0.412, speed=500.0, voltage=6.0, mass=0.022)
kst_ds215mg = Servo("KST DS215MG", torque=1.470, speed=400.0, voltage=6.0, mass=0.065)
