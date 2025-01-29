class hook_features:
    def __init__(self, model, hook_module_names):
        '''
        Parameters
        ----------
        model: pytorch model
        hook_module_names: [str]
            names of modules to be hooked
        '''
        self.myinit()
        self.register_hook(model, hook_module_names)

    def myinit(self):
        self.module_names = []
        self.input_features =  []
        self.output_features =  []

    def register_hook(self, model, hook_module_names):
        for module in model.modules():  # model.children() # only return the direct children
            if self.get_module_name(module) in hook_module_names:
                module.register_forward_hook(hook=self.hook_func)

    def get_module_name(self, module):
        '''
            module: pytorch module
        '''
        return str(module.__class__)[:-2].split('.')[-1] 

    def hook_func(self, module, in_feat, out_feat):
        self.module_names.append(self.get_module_name(module))
        self.input_features.append(in_feat)
        self.output_features.append(out_feat)

    def get_hook_results(self):
        '''
            This function should be called after calling `model.forward`.
        '''
        names = self.module_names[:]
        inputs = self.input_features[:]
        outputs = self.output_features[:]
        self.myinit()
        return names, inputs, outputs
