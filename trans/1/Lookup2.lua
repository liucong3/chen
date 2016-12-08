require 'nn'

local Lookup, parent = torch.class('nn.Lookup', 'nn.Module');

function Lookup:__init(input_count, output_size)
	self.weight = torch.Tensor(input_count, output_size);
	self._gradWeight = {}
	self._input = {}
	self.output = torch.Tensor()
	self.gradInput = torch.Tensor()
end

function Lookup:parameters()
	return {self.weight}, { torch.Tensor() }
end

function Lookup:updateOutput(input)
	local output = self.weight:index(1, input)
	return self.output:set(output)
end

function Lookup:updateGradInput(input, gradOutput)
	return self.gradInput
end

function Lookup:zeroGradParameters()
	local size = #self._gradWeight
	for i = 1, size do
		self._gradWeight[i] = nil
		self._input[i] = nil
	end
end

function Lookup:accGradParameters(input, gradOutput, scale)
	local gradOutput = scale and torch.mul(gradOutput, scale) or gradOutput:clone()
	local size = #self._gradWeight
	self._gradWeight[size + 1] = gradOutput
	self._input[size + 1] = input
end

function Lookup:updateParameters(learningRate)
	for i = 1, #self._gradWeight do
		local gradWeight = self._gradWeight[i]
		gradWeight = torch.mul(gradWeight, -learningRate)
		self.weight:indexAdd(1, self._input[i], gradWeight)
	end
end

function Lookup:accUpdateGradParameters(input, gradOutput, lr)
	self:zeroGradParameters()
	self:accGradParameters(input, gradOutput, 1)
	self:updateParameters(lr)
end

function Lookup:shareClone()
	local lookup = nn.Lookup(1,1)
	lookup.weight = self.weight
	lookup._gradWeight = self._gradWeight
	lookup._input = self._input
	return lookup
end

------------------



--[[
local seq = nn.Sequential()
seq:add(nn.Lookup(4,3))
seq:add(nn.Linear(3,5))
local input = torch.LongTensor({1,3,4})
gradientCheck(seq, input)
--[[]]--

--[[
require 'gradientCheck'
require 'Dynamic'

local Test, parent = torch.class('nn.Test', 'nn.Dynamic')

function Test:__init(lookup, linear)
    self.lookup = lookup or nn.Lookup(3,3)
    self.linear = linear or nn.Linear(3,2)
    parent.__init(self, self.lookup, self.linear)
end

function Test:updateOutput(input)
	self.forwSequence = {}
	local hidden = self:DF(self.lookup, input)
	local output = self:DF(self.linear, hidden)
	self:setOutput(output)
    return self.output
end

function Test:shareClone()
	local test = nn.Test(self.lookup, self.linear)
	return test
end

local test = nn.Test()
local input = torch.LongTensor({1,3,3})
gradientCheck(test, input)
--[[]]--
