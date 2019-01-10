import time
class Analyzer:
	def __init__(self):
		self.info = {}
	def start(self,timer_name):
		if(timer_name in self.info):
			self.info[timer_name][0] = time.time()
		else:
			#[start time,logged times]
			self.info[timer_name] = [time.time(),[]]
	def stop(self, timer_name):
		if(timer_name in self.info):
			if(self.info[timer_name][0] != -1):
				self.info[timer_name][1].append(time.time()-self.info[timer_name][0])
				self.info[timer_name][0] = -1
			else:
				print("Time Analyzer timer ("+str(timer_name)+" has not been started.")
		else:
			print("Time Analyzer timer ("+str(timer_name)+") does not exist.")
	def show_timer(self,timer_name,verbose=False):
		if(timer_name in self.info):
			ttl = sum(self.info[timer_name][1])
			avg = ttl/float(len(self.info[timer_name][1]))
			print("Timer ("+str(timer_name)+"):")
			print("   Average: "+str(avg))
			print("   Total Time: "+str(ttl))
			if(verbose):
				print("   Recorded Values: "+str(self.info[timer_name][1]))
		else:
			print("Time Analyzer timer ("+str(timer_name)+") does not exist.")
	def show_times(self,verbose=False):
		for key, value in self.info.items():
			ttl = sum(value[1])
			avg = ttl/float(len(value[1]))
			print("Timer ("+str(key)+"):")
			print("   Average: "+str(avg))
			print("   Total Time: "+str(ttl))
			print("   Times Executed: "+str(len(value[1])))
			if(verbose):
				print("   Recorded Values: "+str(value[1]))
