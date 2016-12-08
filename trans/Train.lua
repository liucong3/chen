 
require("sys")

local Train = torch.class("Train")

-- Initialization of the trainer class
-- data: the data object
-- model: the model object
-- loss: the loss object
-- config: (optional) the configuration table
--      .rates: (optional) the table of learning rates, indexed by the number of epoches
--      .epoch: (optional) current epoch
function Train:__init(data, model, loss, config)
   -- Store the objects
   self.data = data
   self.model = model
   self.loss = loss

   -- Store the configurations and states
   local config = config or {}
   self.rates = config.rates or {1e-3}
   self.epoch = config.epoch or 1
   self.clamp = config.clamp or 1e6

   -- Get the parameters and gradients
   self.params, self.grads = self.model:getParameters()
   self.params:copy(torch.randn(self.params:size()) * 0.01)
   self.momentum_grads = self.grads:clone():zero() -- momentum
   --self.look_ahead_grads = self.momentum_grads:clone()

   -- Make the loss correct type
   self.loss:type(self.model:type())

   -- Find the current rate
   self:_findTheCurrentRate()

   -- Timing table
   self:clearTime()

   -- Store the configurations
   self.momentum = config.momentum or 0
   self.decay = config.decay or 0
   self.normalize = config.normalize
   self.recapture = config.recapture
   self.collectgarbage = config.collectgarbage or 1
end

function Train:_findTheCurrentRate()
   local max_epoch = 1
   self.rate = self.rates[1]
   for i,v in pairs(self.rates) do
      if i <= self.epoch and i > max_epoch then
         max_epoch = i
         self.rate = v
      end
   end
end

-- Run for a number of steps
-- epoches: number of epoches
-- logfunc: (optional) a function to execute after each step.
function Train:run(epoches, logfunc)
   --[[ Recapture the weights
   if self.recapture then
      self.params,self.grads = nil,nil
      collectgarbage()
      self.params,self.grads = self.model:getParameters()
      collectgarbage()
   end]]--
   -- The loop
   for i = 1,epoches do
      self:batchStep()
      collectgarbage()
      if logfunc then logfunc(self, i) end
   end
end

function Train:clearTime()
   self.time = {}
   self.time.data = 0
   self.time.forward = 0
   self.time.backward = 0
   self.time.update = 0
end

function Train:_recordTime(type)
   if self.model:type() == "torch.CudaTensor" then cutorch.synchronize() end
   if self.model:type() == "torch.ClTensor" then cltorch.synchronize() end
   self.time[type] = self.time[type] + sys.clock() - self.clock
   self.clock = sys.clock()
end

-- Run for one batch step
function Train:batchStep()
   self.clock = sys.clock()

   -- Get a batch of data, make the data to correct type
   self.batch_untyped,self.labels_untyped = self.data:getBatch(self.batch_untyped,self.labels_untyped) 
   self.batch = self.batch or self.batch_untyped:t():contiguous()--:type(self.model:type())
   self.labels = self.labels or self.labels_untyped:t():contiguous()--:type(self.model:type())
   self.batch:copy(self.batch_untyped:t())
   self.labels:copy(self.labels_untyped:t())
   self:_recordTime('data')

   --self.params:add(self.look_ahead_grads)

   -- Forward propagation
   self.output = self.model:forward({self.batch, self.labels})
   self.objective = self.loss:forward(self.output,self.labels)
   if type(self.objective) ~= "number" then self.objective = self.objective[1] end
   self:_recordTime('forward')

   -- Backward propagation   
   self.grads:zero()
   self.gradOutput = self.loss:backward(self.output, self.labels)
   self.gradBatch = self.model:backward(self.batch, self.gradOutput)
   self:_recordTime('backward')

   -- Update: clamp gradients, L2 regularization, momentum
   print(string.format('grad:%.4f/%.4f, momentum:%.4f/%.4f, param:%.4f/%.4f',
      self.grads:norm(), self.grads:max(), self.momentum_grads:norm(), 
      self.momentum_grads:max(), self.params:norm(), self.params:max()))
   
   --self.params:csub(self.look_ahead_grads)

   self.grads:clamp(-self.clamp, self.clamp)
   self.momentum_grads:mul(self.momentum):add(self.grads:mul(-self.rate))
   self.params:mul(1-self.rate*self.decay):add(self.momentum_grads)
   self.params:clamp(-self.clamp, self.clamp)
   self:_recordTime('update')

   --self.look_ahead_grads:copy(self.momentum_grads):mul(self.momentum):add(torch.mul(self.params, -self.rate*self.decay))

   -- Increment on the epoch
   self.epoch = self.epoch + 1
   -- Change the learning rate
   self.rate = self.rates[self.epoch] or self.rate
end
