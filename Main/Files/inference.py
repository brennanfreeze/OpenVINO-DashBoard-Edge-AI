from openvino.inference_engine import IECore, IENetwork

class Network:
    #Constructor class to declare variables, any of these still as 'None' in console, an error occured when initializing it
    def __init__(self):
        #NEED TO: put ntoes done indicating what each does
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None

    def load_model(self, model, bin, device = "CPU"):
        #Brings in IR file to be read as an .xml, morphs string to be seen as a .bin in the same folder, as it should be
        model_xml = model
        model_bin = bin

        self.plugin = IECore()
        self.network = IENetwork(model_xml, weights = model_bin)
        self.exec_network = self.plugin.load_network(self.network, device)

        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        return 

    def get_input_shape(self):
        return self.network.inputs[self.input_blob].shape
    
    def async_inference(self, image):
        self.exec_network.start_async(request_id = 0, inputs = {self.input_blob: image})
        return

    def synchronous_inference(self,image):
        self.exec_network.infer({self.input_blob: image})
        return

    def wait(self):
        status = self.exec_network.requests[0].wait(-1)
        return status

    def extract_output(self):
        return self.exec_network.requests[0].outputs[self.output_blob]



