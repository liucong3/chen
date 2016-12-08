
require 'nn'
require 'Dynamic'
require 'MergeTable'
require 'BranchTable'

--[[
local batch_size = 6
local input_sizes = {3, 4, 5}
local gru = nn.GridGRU(input_sizes)
-- model
local parameters = gru:getParameters()
parameters:copy(torch.randn(parameters:size()))
-- input
local input = {}
for i = 1, #input_sizes do
    input[i] = torch.randn(batch_size, input_sizes[i])
end
-- forward
local output = gru:forward(input)
for i = 1, #output do
    print(output[i])
end
-- parameters
print(gru.Wrz.weight:size())
print(gru.Wrz.bias:size())
print(gru.Wu.weight:size())
print(gru.Wu.bias:size())
]]--

local GridGRU, parent = torch.class('nn.GridGRU', 'nn.Dynamic')

function GridGRU:__init(input_sizes, Wrz, Wu)
    self.input_sizes = input_sizes
    self.sum_input_sizes = 0
    for i = 1, #input_sizes do
        self.sum_input_sizes = self.sum_input_sizes + input_sizes[i]
    end
    self.Wrz = Wrz or nn.Linear(self.sum_input_sizes, 2 * self.sum_input_sizes)
    self.Wu = Wu or nn.Linear(self.sum_input_sizes, self.sum_input_sizes)
    parent.__init(self, self.Wrz, self.Wu)
end

function GridGRU:updateOutput(input)
    self:setInput(unpack(input))
    local input1 = self:DF(nn.MergeTable(#self.input_sizes), unpack(input))
    local rz = self:DF(self.Wrz, input1)
    rz = self:DF(nn.Sigmoid(), rz)
    local r, z = self:DF(nn.BranchTable{self.sum_input_sizes, self.sum_input_sizes}, rz)
    input1 = self:DF(nn.CMulTable(), input1, r)
    local u = self:DF(nn.Tanh(), self:DF(self.Wu, input1))
    local not_z = self:DF(nn.AddConstant(1,false), self:DF(nn.MulConstant(-1,false), z))
    output1 = self:DF(nn.CAddTable(), self:DF(nn.CMulTable(), input1, z), self:DF(nn.CMulTable(), u, not_z))
    local output = { self:DF(nn.BranchTable(self.input_sizes), output1) }
    return self:setOutput(unpack(output))
end

function GridGRU:shareClone()
    local lstm = nn.GridGRU(self.input_sizes, self.Wrz, self.Wu)
    return lstm
end

-------

--[[
require 'gradientCheck'

local input_sizes = {3, 4, 5}
gru = nn.GridGRU(input_sizes)
local input_size = 0
for i = 1, #input_sizes do
    input_size = input_size + input_sizes[i]
end
gradientCheck(gru, {5, input_size}, input_sizes)
--[[]]--




