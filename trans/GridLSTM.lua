
require 'nn'
require 'Dynamic'
require 'MergeTable'
require 'BranchTable'

--[[
local batch_size = 6
local input_sizes = {3, 4, 5}
local lstm = nn.GridLSTM(input_sizes)
-- model
local parameters = lstm:getParameters()
parameters:copy(torch.randn(parameters:size()))
-- input
local input = {}
for i = 1, #input_sizes do
    input[i] = torch.randn(batch_size, input_sizes[i]) -- cell
    input[#input_sizes + i] = torch.randn(batch_size, input_sizes[i]) -- hidden
end
-- forward
local output = lstm:forward(input)
for i = 1, #output do
    print(output[i])
end
-- parameters
print(lstm.W.weight:size())
print(lstm.W.bias:size())
]]--

local GridLSTM, parent = torch.class('nn.GridLSTM', 'nn.Dynamic')

function GridLSTM:__init(input_sizes, W)
    self.input_sizes = input_sizes
    self.sum_input_sizes = 0
    for i = 1, #input_sizes do
        self.sum_input_sizes = self.sum_input_sizes + input_sizes[i]
    end
    self.W = W or nn.Linear(self.sum_input_sizes, 4 * self.sum_input_sizes)
    parent.__init(self, self.W)
end

function GridLSTM:updateOutput(input)
	self:setInput(unpack(input))
    local c1 = {}
    local h1 = {}
    assert(#input == 2 * #self.input_sizes, '#input is ' .. #input .. ', should be ' .. 2 * #self.input_sizes)
    for i = 1, #self.input_sizes do
        c1[i] = input[i]
        h1[i] = input[#self.input_sizes + i]
    end
    c1 = self:DF(nn.MergeTable(#self.input_sizes), unpack(c1))
    h1 = self:DF(nn.MergeTable(#self.input_sizes), unpack(h1))
    local a = self:DF(self.W, h1)
	local hs = self.sum_input_sizes;
    local u, ifo = self:DF(nn.BranchTable{hs, 3 * hs}, a)
    u = self:DF(nn.Tanh(), u)
    ifo = self:DF(nn.Sigmoid(), ifo)
	local i, f, o = self:DF(nn.BranchTable{hs, hs, hs}, ifo)
    local c_1 = self:DF(nn.CMulTable(), c1, f)
    local c_2 = self:DF(nn.CMulTable(), u, i)
    local c = self:DF(nn.CAddTable(), c_1, c_2)
    local h = self:DF(nn.CMulTable(), c, o)
    local output = { self:DF(nn.BranchTable(self.input_sizes), c) }
    appendToTable(output, { self:DF(nn.BranchTable(self.input_sizes), h) })
    return self:setOutput(unpack(output))
end

function GridLSTM:shareClone()
	local lstm = nn.GridLSTM(self.input_sizes, self.W)
	return lstm
end

--[[
require 'Check'
require 'MSELoss'

local input_sizes = {3, 4, 5}
local c = Check()
local input = c.randData({{7,3},{7,4},{7,5},{7,3},{7,4},{7,5}})
local label = c.randData({{7,3},{7,4},{7,5},{7,3},{7,4},{7,5}})

local loss = nn.MSELoss()

local T, parent2 = torch.class('Test', 'nn.Dynamic')
function T:__init()
    self.lstm1 = nn.GridLSTM(input_sizes)
    --self.lstm2 = nn.GridLSTM(input_sizes)
    parent.__init(self, self.lstm1)
end
function T:updateOutput(input)
    self:setInput(unpack(input))
    local hidden = {self:DF(self.lstm1, unpack(input))}
    local zeros = {}
    for i = 1, #hidden / 2 do
        zeros[i] = hidden[i]:clone():zero()
    end
    local hidden = {self:DF(self.lstm1, hidden[1], hidden[2], hidden[3], unpack(zeros))}
    return self:setOutput(unpack(hidden))
end

local model = Test()
c.gradCheck(model, loss, input, label, 1e-6)
--[[]]--

--[[
require 'gradientCheck'

local input_sizes = {3, 4, 5}
lstm = nn.GridLSTM(input_sizes)
local input_sizes2 = { unpack(input_sizes) }
for i = 1, #input_sizes do
    table.insert(input_sizes2, input_sizes[i])
end
local input_size = 0
for i = 1, #input_sizes2 do
    input_size = input_size + input_sizes2[i]
end
gradientCheck(lstm, {10, input_size}, input_sizes2)
--[[]]--

--[[
require 'gradientCheck'

local Test, parent2 = torch.class('nn.Test', 'nn.Dynamic')

function Test:__init()
    self.lstm = nn.GridLSTM({4,3})
    parent.__init(self, self.lstm)
end

function Test:updateOutput(input)
    self.forwSequence = {}
    local batchSize = input:size(1)
    local zero4 = torch.zeros(batchSize, 4)
    local zero3 = torch.zeros(batchSize, 3)
    local cell4, cell3, hidden4, hidden3 = self:DF(self.lstm, zero4, zero3, zero4, input)
    return self:setOutput(hidden3)
end

gradientCheck(nn.Test(), {10, 3})
--[[]]--


