
require 'nn'
require 'fixes'

local Dynamic, parent = torch.class('nn.Dynamic', 'nn.Container')

-- must registrate parametric modules in ...
function Dynamic:__init(...)
    parent.__init(self)
    self.modules = {...}
    self.moduleIndexes = {}
    for i, v in ipairs(self.modules) do
        self.moduleIndexes[v] = i
    end
end

function Dynamic:options(...)
    -- hidden shared parametric modules
    local hiddenModules = {...}
    for i, v in ipairs(hiddenModules) do
        self.moduleIndexes[v] = -i
    end
end

-- must set self.forwSequence = {} first thing in updateOutput
function Dynamic:updateOutput(input)
	self:setInput(type(input) == 'table' and unpack(input) or input)
    -- TODO
    error('Must be overrided.')
    return self:setOutput(torch.Tensor()) -- Will raise an exception here
end

function Dynamic:setInput(...)
    self._input = {...}
    self.forwSequence = {}
end

function Dynamic:setOutput(...)
    self._output = {...}
    self.output = (#self._output == 1 and self._output[1] or self._output)
    return self.output
end

-- dynamic forward
function Dynamic:DF(module, ...)
    assert(torch.isTypeOf(module, 'nn.Module'), 'First parameter must be an nn.Module')
    local input = {...}
    if #input == 1 then input = input[1] end
	if module:parameters() then
        local index = self.moduleIndexes[module]
		if not index then
            error('Parametric modules ' .. type(module) .. ' must be registrated with parent.__init(...) or self:options(...)')
        end
 		module = module:shareClone()
	end
    local output = module:updateOutput(input)
    self.forwSequence[#self.forwSequence + 1] = { module, input, output }
    if type(output) == 'table' then
		return unpack(output)
	else
		return output
	end
end

function Dynamic._getGrad(dataToGrad, data)
    if type(data) == 'table' then
        local grad = {}
        local nonEmptyGrads = 0
        for i = 1, #data do
            grad[i] = dataToGrad[data[i]]
            if grad[i] then nonEmptyGrads = nonEmptyGrads + 1 end
        end
        if nonEmptyGrads == 0 then return nil end
        return grad
    else
        return dataToGrad[data]
    end
end

function Dynamic._addGrad(dataToGrad, data, grad)
    if type(data) == 'table' then
        for i = 1, #data do
            Dynamic._addGrad(dataToGrad, data[i], grad[i])
        end
    else
        if not dataToGrad[data] then
            dataToGrad[data] = grad
        else
            dataToGrad[data]:add(grad)
        end
    end
end

function Dynamic._removeGrad(dataToGrad, data, grad)
    if type(data) == 'table' then
        for i = 1, #data do
            Dynamic._removeGrad(dataToGrad, data[i], grad[i])
        end
    else
        dataToGrad[data] = nil
    end
end

function Dynamic._getAddGrad(dataToGrad, output)
	local gradOutput = Dynamic._getGrad(dataToGrad, output)
	if type(gradOutput) ~= 'table' then return gradOutput end
	for i = 1, #output do
		if not gradOutput[i] then
			local grad = output[i].new():resizeAs(output[i]):zero()
			dataToGrad[output[i]] = grad
			gradOutput[i] = grad
		end
	end
	return gradOutput
end

-- must be called after Dynamic:updateOutput due to self.forwSequence
function Dynamic:updateGradInput(input, gradOutput, scale)
    self.dataToGrad = {}
    if type(gradOutput) ~= 'table' then gradOutput = {gradOutput} end
    Dynamic._addGrad(self.dataToGrad, self._output, gradOutput)

    for i = #self.forwSequence, 1, -1 do
        local module, input2, output2 = unpack(self.forwSequence[i])
        local gradOutput2 = Dynamic._getAddGrad(self.dataToGrad, output2)
		if gradOutput2 then
			--self.forwSequence[i][4] = gradOutput2
		    local gradInput2 = module:backward(input2, gradOutput2, scale)
            Dynamic._removeGrad(self.dataToGrad, output2, gradOutput2) -- to save memory
            Dynamic._addGrad(self.dataToGrad, input2, gradInput2)
            self.forwSequence[i] = nil -- to save memory
		end
        self:_collectgarbage()
    end

    self.gradInput = Dynamic._getGrad(self.dataToGrad, input)
    self.dataToGrad = nil -- to save memory
    return self.gradInput
end

-- must be called after Dynamic:updateGradInput due to self.dataToGrad
function Dynamic:accGradParameters(input, gradOutput, scale)
    error('Error: Not supported.\nIn nn.Dynamic, accGradParameters is combined with updateGradInput in order to save memory.')
	-- for i = #self.forwSequence, 1, -1 do
    --      local module, input2, output2, gradOutput2 = unpack(self.forwSequence[i])
	-- 	if gradOutput2 then
	-- 	    module:accGradParameters(input2, gradOutput2, scale)
	-- 	end
	-- end
end

function Dynamic:backward(input, gradOutput, scale)
   scale = scale or 1
   self:updateGradInput(input, gradOutput, scale)
   return self.gradInput
end 

function Dynamic:reset(stdv)
    parent.reset(self)
    self.forwSequence = {}
    self.dataToGrad = {}
end

function Dynamic:clearState()
    parent.clearState(self)
    self.forwSequence = {}
    self.dataToGrad = {}
end

function Dynamic:_collectgarbage()
    if not self._collectgarbage_clock then
        self._collectgarbage_interval = 5
        self._collectgarbage_clock = sys.clock()
        return
    end
    local clock = sys.clock()
    local duration = clock - self._collectgarbage_clock
    if duration > self._collectgarbage_interval then
        self._collectgarbage_clock = clock
        print('collectgarbage')
        collectgarbage()
        print('collectgarbage done')
    end
end

--[[


require 'gradientCheck'

local Test, parent = torch.class('nn.Test', 'nn.Dynamic')

function Test:__init(input_size)
    self.linear = nn.Linear(input_size, input_size)
    parent.__init(self, self.linear)
end

function Test:updateOutput(input)
	self.forwSequence = {}
    local output = self:DF(self.linear, input)
    --print(output)
    return self:setOutput(output)
end

local test = nn.Test(5)
gradientCheck(test, {7, 5})

--[[]]--

