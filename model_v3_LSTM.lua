equire('nngraph')
require('nn')
require('graph')
require('os')


function getStepModule(width, height)
	-- Inputs to each step of the network --
	local xx = - nn.Identity()
	local h0 = - nn.Identity()
	local c0 = - nn.Identity()
	local aff_matrix = - nn.Identity()

	-- Breaking compound hidden state into 3 component states --
	local h01 = h0 - nn.Narrow(1, 1, 16)
	local h02 = h0 - nn.Narrow(1, 17, 16)
	local h03 = h0 - nn.Narrow(1, 33, 16)

	-- Breaking compound cell state into 3 component states
	local c01 = c0 - nn.Narrow(1, 1, 16)
	local c02 = c0 - nn.Narrow(1, 17, 16)
	local c03 = c0 - nn.Narrow(1, 33, 16)

	-- LSTM_1 : Hidden State 1 --
	local preactivations = {xx, h01} - nn.JoinTable(1) - nn.SpatialDilatedConvolution(18, 4*16, 3, 3, 1, 1, 1, 1, 1, 1)

	local all_gates = preactivations - nn.Narrow(1, 1, 3*16) - nn.Sigmoid()
	local in_transform = preactivations - nn.Narrow(1, (3*16 + 1), 16) - nn.Tanh()

	local in_gate = all_gates - nn.Narrow(1, 1, 16)
	local forget_gate = all_gates - nn.Narrow(1, (16+1), 16)
	local out_gate = all_gates - nn.Narrow(1, (2*16 + 1), 16)

	local c_forget = {forget_gate, c01} - nn.CMulTable()
	local c_input = {in_gate, in_transform} - nn.CMulTable()
	local c11 = {c_forget, c_input} - nn.CAddTable()

	local c_transform = c11 - nn.Tanh()
	local h11 = {out_gate, c_transform} - nn.CMulTable()

	-- LSTM_2 : Hidden State 2 --
	local preactivations = {h11, h02} - nn.JoinTable(1) - nn.SpatialDilatedConvolution(32, 4*16, 3, 3, 1, 1, 2, 2, 2, 2)

	local all_gates = preactivations - nn.Narrow(1, 1, 3*16) - nn.Sigmoid()
	local in_transform = preactivations - nn.Narrow(1, (3*16 + 1), 16) - nn.Tanh()

	local in_gate = all_gates - nn.Narrow(1, 1, 16)
	local forget_gate = all_gates - nn.Narrow(1, (16+1), 16)
	local out_gate = all_gates - nn.Narrow(1, (2*16 + 1), 16)

	local c_forget = {forget_gate, c02} - nn.CMulTable()
	local c_input = {in_gate, in_transform} - nn.CMulTable()
	local c12 = {c_forget, c_input} - nn.CAddTable()

	local c_transform = c12 - nn.Tanh()
	local h12 = {out_gate, c_transform} - nn.CMulTable()

	-- LSTM_3 : Hidden State 3 --
	local preactivations = {h12, h03} - nn.JoinTable(1) - nn.SpatialDilatedConvolution(32, 4*16, 3, 3, 1, 1, 4, 4, 4, 4)

	local all_gates = preactivations - nn.Narrow(1, 1, 3*16) - nn.Sigmoid()
	local in_transform = preactivations - nn.Narrow(1, (3*16 + 1), 16) - nn.Tanh()

	local in_gate = all_gates - nn.Narrow(1, 1, 16)
	local forget_gate = all_gates - nn.Narrow(1, (16+1), 16)
	local out_gate = all_gates - nn.Narrow(1, (2*16 + 1), 16)

	local c_forget = {forget_gate, c03} - nn.CMulTable()
	local c_input = {in_gate, in_transform} - nn.CMulTable()
	local c13 = {c_forget, c_input} - nn.CAddTable()

	local c_transform = c13 - nn.Tanh()
	local h13 = {out_gate, c_transform} - nn.CMulTable()

	-- Combining component cell and hidden states --
	local h1 = {h11, h12, h13} - nn.JoinTable(1)
	local c1 = {c11, c12, c13} - nn.JoinTable(1)

	local yy = h1 - nn.SpatialConvolution(48, 1, 3, 3, 1, 1, 1, 1) - nn.Sigmoid()
	
	return nn.gModule({h0, c0, xx}, {h1, c1, yy})
end
