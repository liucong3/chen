local G = torch.class('Check')

function G.gradCheck(module, criterion, input, label, delta)
    delta = delta or 1e-6
    local grad_param, grad_input = G.analyticalGrad(module, criterion, input, label)

    print('\n-- Module --')
    G.printModule(module, input, 1, '')
    print('\n-- Params --')
    G.printParams(module)

    local paramNames = G.getParams(module, '')
    if not delta or delta == 0 then
        print('\n-- Grad params --')
        G.printAbsMean(paramNames, grad_param)
        print('\n-- Grad input --')
        G.printAbsMean('grad_input', grad_input)
    else
        local grad_param2, grad_input2 = G.empiricalGrad(module, criterion, input, label, delta)
        if grad_param then
            print('\n-- Error grad param --')
            G.compare(paramNames, grad_param, grad_param2)
        end
        print('\n-- Error grad input --')
        G.compare('input', grad_input, grad_input2)
        --print('--->', grad_input, grad_input2)
    end

    print('')
end
   
function G.randParams(module, var)
    local params = module:parameters()
    for i = 1, #params do
        local rand = torch.randn(params[i]:size())
        if var then rand = rand * var end
        params[i]:copy(rand)
    end
end

function G.randData(sizes, var)
    local input = {}
    for i = 1, #sizes do
        input[i] = torch.randn(unpack(sizes[i]))
        if var then input[i] = input[i] * var end
    end
    return input
end

function G.randIndex(sizes, ranges)
    local input = {}
    for i = 1, #sizes do
        input[i] = (torch.rand(unpack(sizes[i])) * ranges[i]) + 1
        input[i] = input[i]:long()
    end
    return input
end

function G.getParams(module, recursive)
    local names = {}
    local params = {}

    local function findKey(map, elem)
        for key, val in pairs(map) do
            if val == elem then return key end
        end
        return nil
    end

    local function toMap(array, map, ignore, names, data)
        if not array then return {}, {} end
        names = names or {}
        data = data or {}
        for i = 1, #array do
            local key = findKey(map, array[i])
            if key or not ignore then
                if not key then key = '#' .. i end
                names[#names + 1] = key
                data[#data + 1] = array[i]
            end
        end
        return names, data
    end

    local function forSubModules(prefix, module)
        if torch.isTypeOf(module, 'nn.Dynamic') then
            local names, submodule = toMap(module.modules, module)
            for i = 1, #names do
                forSubModules(prefix .. '.' .. names[i], submodule[i])
            end
        else
            local names2, params2 = toMap(module:parameters(), module)
            for i = 1, #names2 do
                names[#names + 1] = prefix .. '.' .. names2[i]
                params[#params + 1] = params2[i]
            end
        end
    end

    recursive = (recursive == nil and '' or recursive)
    if recursive then
        forSubModules(recursive, module)
        return names, params
    else
        local params = module:parameters()
        return toMap(params, module, true)
    end
end

function G.printModule(module, input, depth, line_prefix)
    line_prefix = line_prefix or ''
    if not torch.isTypeOf(module, 'nn.Dynamic') then
        local paramNames = G.getParams(module, false)
        paramNames = (#paramNames == 0 and '' or '[' .. table.concat(paramNames, ',') .. ']')
        print(string.format('%s%s%s', line_prefix, torch.type(module), paramNames))
        return
    end

    if not module.forwSequence then
        assert(input ~= nil, 'Must call forward before printModule for nn.Dynamic.')
        module:forward(input)
    end

    local dataIdMap = {}
    local dataCount = 0

    local function getIds(data, ids)
        ids = ids or {}
        if type(data) == 'table' then
            for i = 1, #data do
                getIds(data[i], ids)
            end
        else
            if not dataIdMap[data] then
                dataCount = dataCount + 1
                dataIdMap[data] = dataCount
            end
            ids[#ids + 1] = dataIdMap[data]
        end
        return ids
    end

    local tab = '   '
    print(string.format('%s<x%d>:\t%s -> %s', line_prefix, #module.forwSequence, 
        table.concat(getIds(module._input), ','), 
        table.concat(getIds(module._output), ',')))

    for i = 1, #module.forwSequence do
        local submodule, input, output = unpack(module.forwSequence[i])
        local paramNames = G.getParams(submodule, false)
        paramNames = (#paramNames == 0 and '' or '[' .. table.concat(paramNames, ',') .. ']')
        print(string.format('%s%s%s:\t%s -> %s', line_prefix, torch.type(submodule), 
            paramNames, table.concat(getIds(input), ','), table.concat(getIds(output), ',')))
        if torch.isTypeOf(submodule, 'nn.Dynamic') and depth ~= 1 then
            if depth then depth = depth - 1 end
            G.printModule(submodule, nil, depth, line_prefix .. tab)
        end
    end
end

function G.printAbsMean(names, data, digits)
    digits = digits or 6

    local function printAbsMean1(name, data1)
        if not data1 then
            print(name .. ': nil')
        elseif ##data1 == 0 then
            print(name .. ': emtpy')
        else
            print(string.format('%s[%s]\tabs_mean=%.' .. digits .. 'f', 
                name, table.concat(torch.totable(#data1), ','), torch.abs(data1):mean()))
        end
    end

    if type(data) ~= 'table' then
        printAbsMean1(names, data)
    else
        if type(names) ~= 'table' then
            local name = names
            names = {}
            for i = 1, #data do
                names[i] = name .. '[' .. i .. ']'
            end
        end
        for i = 1, #names do
            printAbsMean1(names[i], data[i])
        end
    end
end

function G.printParams(module)
    local names, params = G.getParams(module)
    G.printAbsMean(names, params)
end

function G.analyticalGrad(module, criterion, input, label)
    local params, gradParams = module:parameters()
    --module:zeroGradParameters()
    if gradParams then
        for i = 1, #gradParams do gradParams[i]:zero() end
    end
    local output = module:forward(input)
    --print('--->\n', params[1], params[2], output)
    criterion:forward(output, label)
    local gradOutput = criterion:backward(output, label)
    local gradInput = module:backward(input, gradOutput)
    return gradParams, gradInput
end

function G.empiricalGrad(module, criterion, input, label, delta)

    local function empiricalGrad1(module, criterion, input, label, delta, var)
        local output = module:forward(input)
        local cost = criterion:forward(output, label)
        local grad = var.new(var:size())
        local grad1 = grad:view(-1)
        local var1 = var:view(-1)
        for i = 1, var1:size(1) do
            local original = var1[i]
            var1[i] = original + delta
            local output2 = module:forward(input)
            grad1[i] = (criterion:forward(output2, label) - cost) / delta
            var1[i] = original
        end
        return grad
    end

    local params = module:parameters()

    local gradParams = nil
    if params then
        gradParams = {}
        for i = 1, #params do
            gradParams[i] = empiricalGrad1(module, criterion, input, label, delta, params[i])
        end
    end

    local _, gradInput1 = G.analyticalGrad(module, criterion, input, label)
    local gradInput = nil
    if gradInput1 then
        if type(input) == 'table' then
            gradInput = {}
            for i = 1, #input do
                if gradInput1[i] then
                    gradInput[i] = empiricalGrad1(module, criterion, input, label, delta, input[i])
                end
            end
        else
            gradInput = empiricalGrad1(module, criterion, input, label, delta, input)
        end
    end

    return gradParams, gradInput
end

function G.compare(names, analyticGrad, empiricalGrad)
    local diff = nil
    if type(analyticGrad) == 'table' then
        diff = {}
        for i = 1, #analyticGrad do
            if analyticGrad[i] then
                diff[i] = analyticGrad[i] - empiricalGrad[i]
            end
        end
    elseif analyticGrad then
        diff = analyticGrad - empiricalGrad
    end
    G.printAbsMean(names, diff, 15)
end

--[[
require 'Dynamic'
local c = Check()
M, p = torch.class('Test', 'nn.Dynamic')
function M:__init(linear1, linear2)
    self.linear1 = linear1 or nn.Linear(3,4)
    self.linear2 = linear2 or nn.Linear(4,3)
    p.__init(self, self.linear1, self.linear2)
end
function M:updateOutput(input)
    self:setInput(input)
    local hidden = self:DF(self.linear1, input)
    hidden = self:DF(nn.Tanh(), hidden)
    hidden = self:DF(self.linear2, hidden)
    hidden = self:DF(nn.SoftMax(), hidden)
    return self:setOutput(hidden)
end
local linear1 = nn.Linear(3,4)
local linear2 = nn.Linear(4,3)
--print('==>\n', linear1.weight, linear1.bias)
local model = Test(linear1, linear2)
local input = c.randData({{1,3}})[1]
local label = c.randData({{1,3}})[1]
local criterion = nn.MSECriterion()
c.gradCheck(model, criterion, input, label, 1e-6)
--print('==>\n', linear1.weight, linear1.bias)

local seq = nn.Sequential()
seq:add(linear1)
seq:add(nn.Tanh())
seq:add(linear2)
seq:add(nn.SoftMax())
c.gradCheck(seq, criterion, input, label, 1e-6)
--print(model:forward(input))
--print(model:forward(input))
--[[]]--
