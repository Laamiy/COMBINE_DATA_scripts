import logging 
#Create a custom logger for better output debuging .
Logger = logging.Logger("custom_logger")
#Setting the default root logger minimum log level :
logging.getLogger().setLevel("DEBUG")
console_log_handler = logging.StreamHandler()
#OUTPUT format for each log (nearly the same as print)  : 
custom_formatter    = logging.Formatter("[%(module)-18s] - [%(levelname)-6s] - [%(funcName)-10s] : [ %(message)s ]")
#Apply the format on the handler :
console_log_handler.setFormatter(custom_formatter)
#Set minimum log level : 
console_log_handler.setLevel("DEBUG")
# Attach the console_log_handler to the custom_logger: 
Logger.addHandler(console_log_handler)
