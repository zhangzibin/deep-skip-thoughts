require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'
local BatchLoader = require 'util.BatchLoader'
local model_utils = require 'util.model_utils'
local GRU = require 'model.GRU'

cmd = torch.CmdLine()
cmd:text()
cmd:text('RNN-RNN for Skip Thoughts')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','book_corpus_small/','data directory. Should contain the file input.txt with input data')
-- model params
cmd:option('-seq_length',30,'number of timesteps to unroll for')
cmd:option('-dim_word',500,'word vector dimensionality')
cmd:option('-encoder_forw_rnn_size',1000,'size of RNN encoder forward internal state')
cmd:option('-encoder_back_rnn_size',1000,'size of RNN encoder backward internal state')
cmd:option('-encoder_num_layers',2,'number of layers in the RNN encoder')
cmd:option('-decoder_rnn_size',1000,'size of RNN decoder internal state')
cmd:option('-decoder_num_layers',2,'number of layers in the RNN decoder')
cmd:option('-min_freq',100,'min frequency of words in vocabulary')
-- optimization
cmd:option('-learning_rate',0.005,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',5,'in number of epochs, when to start decaying the learning rate')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer of encoder. 0 = no dropout')
cmd:option('-batch_size',50,'number of sequences to train on in parallel')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',10000,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
-- train / val split for data, in fractions
local split_sizes = {math.max(0, 1 - opt.val_frac), opt.val_frac} 
-- the size of sentence vector is the sum of rnn unit in last layer
opt.vec_size = opt.encoder_forw_rnn_size + opt.encoder_back_rnn_size

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 then
    local ok, cunn = pcall(require, 'cunn')
    if ok then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('Package cunn not found! If cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- create the data loader class
local loader = BatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes, opt.min_freq)
local vocab_size = loader.vocab_size  -- the number of words in vocabulary
local vocab = loader.vocab_mapping
print('vocab size: ' .. vocab_size)
-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

-- create the model, encoder and then decoder
protos = {}
print('creating encoder with ' .. opt.encoder_num_layers .. ' layers')
protos.encoder_forw = GRU.gru(opt.dim_word, opt.encoder_forw_rnn_size, opt.encoder_num_layers, opt.dropout, 0, vocab_size)
protos.encoder_back = GRU.gru(opt.dim_word, opt.encoder_back_rnn_size, opt.encoder_num_layers, opt.dropout, 0, vocab_size)
print('creating decoder with ' .. opt.decoder_num_layers .. ' layers')
protos.decoder_prev = GRU.gru(opt.dim_word, opt.decoder_rnn_size, opt.decoder_num_layers, opt.dropout, opt.vec_size, vocab_size)
protos.decoder_next = GRU.gru(opt.dim_word, opt.decoder_rnn_size, opt.decoder_num_layers, opt.dropout, opt.vec_size, vocab_size)
print('creating critterion')
protos.criterion_prev = nn.ClassNLLCriterion()
protos.criterion_next = nn.ClassNLLCriterion()

-- the initial state of the hidden states
encoder_forw_init_state = {}
encoder_back_init_state = {}
decoder_prev_init_state = {}
decoder_next_init_state = {}
for L=1,opt.encoder_num_layers do
    local encoder_forw_h_init = torch.zeros(opt.batch_size, opt.encoder_forw_rnn_size)
    local encoder_back_h_init = torch.zeros(opt.batch_size, opt.encoder_back_rnn_size)
    if opt.gpuid >=0 then 
        encoder_forw_h_init = encoder_forw_h_init:cuda() 
        encoder_back_h_init = encoder_back_h_init:cuda() 
    end
    table.insert(encoder_forw_init_state, encoder_forw_h_init:clone())
    table.insert(encoder_back_init_state, encoder_back_h_init:clone())
end
for L=1,opt.decoder_num_layers do
    local decoder_prev_h_init = torch.zeros(opt.batch_size, opt.decoder_rnn_size)
    local decoder_next_h_init = torch.zeros(opt.batch_size, opt.decoder_rnn_size)
    if opt.gpuid >=0 then 
        decoder_prev_h_init = decoder_prev_h_init:cuda() 
        decoder_next_h_init = decoder_next_h_init:cuda() 
    end
    table.insert(decoder_prev_init_state, decoder_prev_h_init:clone())
    table.insert(decoder_next_init_state, decoder_next_h_init:clone())
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

-- put the above things into one flattened parameters tensor
encoder_forw_params, encoder_forw_grad_params = model_utils.combine_all_parameters(protos.encoder_forw)
encoder_back_params, encoder_back_grad_params = model_utils.combine_all_parameters(protos.encoder_back)
decoder_prev_params, decoder_prev_grad_params = model_utils.combine_all_parameters(protos.decoder_prev)
decoder_next_params, decoder_next_grad_params = model_utils.combine_all_parameters(protos.decoder_next)

-- initialization
if do_random_init then
    encoder_forw_params:uniform(-0.08, 0.08) -- small uniform numbers
    encoder_back_params:uniform(-0.08, 0.08) -- small uniform numbers
    decoder_prev_params:uniform(-0.08, 0.08) -- small uniform numbers
    decoder_next_params:uniform(-0.08, 0.08) -- small uniform numbers
end

local tol_params_num = encoder_forw_params:nElement() + encoder_back_params:nElement()
                     + decoder_prev_params:nElement() + decoder_next_params:nElement()
print('number of parameters in the model: ' .. tol_params_num)

-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

-- preprocessing helper function
function prepro(x,y_prev,y_next)
    x = x:transpose(1,2):contiguous() -- swap the axes for faster indexing
    y_prev = y_prev:transpose(1,2):contiguous()
    y_next = y_prev:transpose(1,2):contiguous()
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y_prev = y_prev:float():cuda()
        y_next = y_prev:float():cuda()
    end
    return x,y_prev,y_next
end

-- update params
function update(params, grad_params)
    grad_params:div(opt.seq_length)
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    params:add(grad_params:mul(-lr))
end

-- a full forward & backward pass
function myfeval()
    ------------------ clear the gradients -------------
    encoder_forw_grad_params:zero()
    encoder_back_grad_params:zero()
    decoder_prev_grad_params:zero()
    decoder_next_grad_params:zero()
    ------------------ get minibatch -------------------
    local x,y_prev,y_next = loader:next_batch(1)
    x,y_prev,y_next = prepro(x,y_prev,y_next)
    --------------- forward pass encoder ---------------
    local encoder_forw_state = {[0] = encoder_forw_init_state}
    local encoder_back_state = {[0] = encoder_back_init_state}
    for t=1,opt.seq_length do
        clones.encoder_forw[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.encoder_forw[t]:forward{x[t], unpack(encoder_forw_state[t-1])}
        encoder_forw_state[t] = {}
        for i=1,#encoder_forw_init_state do table.insert(encoder_forw_state[t], lst[i]) end -- extract the state, without output
    end
    for t=1,opt.seq_length do
        clones.encoder_back[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.encoder_back[t]:forward{x[opt.seq_length-t+1], unpack(encoder_back_state[t-1])}
        encoder_back_state[t] = {}
        for i=1,#encoder_back_init_state do table.insert(encoder_back_state[t], lst[i]) end -- extract the state, without output
    end
    local sent_vec = torch.cat(encoder_forw_state[opt.seq_length][1], encoder_back_state[opt.seq_length][1])
    --------------- forward pass decoder ---------------
    local decoder_prev_state = {[0] = decoder_prev_init_state}
    local decoder_next_state = {[0] = decoder_next_init_state}
    local predictions_prev = {}           -- softmax outputs
    local predictions_next = {}           -- softmax outputs
    local loss_prev = 0
    local loss_next = 0
    for t=1,opt.seq_length do
        clones.decoder_prev[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.decoder_prev[t]:forward{sent_vec, x[t], unpack(decoder_prev_state[t-1])}
        decoder_prev_state[t] = {}
        for i=1,#decoder_prev_init_state do table.insert(decoder_prev_state[t], lst[i]) end -- extract the state, without output
        predictions_prev[t] = lst[#lst] -- last element is the prediction
        loss_prev = loss_prev + clones.criterion_prev[t]:forward(predictions_prev[t], y_prev[t])
    end
    loss_prev = loss_prev / opt.seq_length
    for t=1,opt.seq_length do
        clones.decoder_next[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.decoder_next[t]:forward{sent_vec, x[t], unpack(decoder_next_state[t-1])}
        decoder_next_state[t] = {}
        for i=1,#decoder_next_init_state do table.insert(decoder_next_state[t], lst[i]) end -- extract the state, without output
        predictions_next[t] = lst[#lst] -- last element is the prediction
        loss_next = loss_next + clones.criterion_next[t]:forward(predictions_next[t], y_next[t])
    end
    loss_next = loss_next / opt.seq_length
    ------------------ backward pass decoder -------------------
    -- gradients on sentence vector
    local dvec = nil
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local d_decoder_prev_state = {[opt.seq_length] = clone_list(decoder_prev_init_state, true)} -- true also zeros the clones
    local d_decoder_next_state = {[opt.seq_length] = clone_list(decoder_next_init_state, true)} -- true also zeros the clones
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion_prev[t]:backward(predictions_prev[t], y_prev[t])
        table.insert(d_decoder_prev_state[t], doutput_t)
        local dlst = clones.decoder_prev[t]:backward({x[t], unpack(decoder_prev_state[t-1])}, d_decoder_prev_state[t])
        d_decoder_prev_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k == 1 then -- k=1 is gradient on sentence vector
                if dvec == nil then dvec = v
                else dvec:add(v) end
            end 
            -- k == 2 is gradient on x, skip
            if k > 2 then d_decoder_prev_state[t-1][k-2] = v end
        end
    end
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion_next[t]:backward(predictions_next[t], y_next[t])
        table.insert(d_decoder_next_state[t], doutput_t)
        local dlst = clones.decoder_next[t]:backward({x[t], unpack(decoder_next_state[t-1])}, d_decoder_next_state[t])
        d_decoder_next_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k == 1 then -- k=1 is gradient on sentence vector
                dvec:add(v) 
            end 
            -- k == 2 is gradient on x, skip
            if k > 2 then d_decoder_next_state[t-1][k-2] = v end
        end
    end
    ------------------ backward pass encoder -------------------
    local d_encoder_forw_state = {[opt.seq_length] = clone_list(encoder_forw_init_state, true)} -- true also zeros the clones
    local d_encoder_back_state = {[opt.seq_length] = clone_list(encoder_back_init_state, true)} -- true also zeros the clones
    d_encoder_forw_state[opt.seq_length][#encoder_forw_init_state] = dvec[{{}, {1,opt.encoder_forw_rnn_size}}]:clone()
    d_encoder_back_state[opt.seq_length][#encoder_back_init_state] = dvec[{{}, {opt.encoder_forw_rnn_size+1,opt.vec_size}}]:clone()
    for t=opt.seq_length,1,-1 do
        local dlst = clones.encoder_forw[t]:backward({x[t], unpack(encoder_forw_state[t-1])}, d_encoder_forw_state[t])
        d_encoder_forw_state[t-1] = {}
        for k,v in pairs(dlst) do
            -- k == 1 is gradient on x, skip
            if k > 1 then d_encoder_forw_state[t-1][k-1] = v end
        end
    end
    for t=opt.seq_length,1,-1 do
        local dlst = clones.encoder_back[t]:backward({x[opt.seq_length-t+1], unpack(encoder_back_state[t-1])}, d_encoder_back_state[t])
        d_encoder_back_state[t-1] = {}
        for k,v in pairs(dlst) do
            -- k == 1 is gradient on x, skip
            if k > 1 then d_encoder_back_state[t-1][k-1] = v end
        end
    end
    ------------------------ misc ----------------------
    -- update the grad_params
    update(encoder_forw_params, encoder_forw_grad_params)
    update(encoder_back_params, encoder_back_grad_params)
    update(decoder_prev_params, decoder_prev_grad_params)
    update(decoder_next_params, decoder_next_grad_params)
    return (loss_prev+loss_next)/2
end

function eval_split(split_index, max_batches)
    print('evaluating loss over split index ' .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end
    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    
    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
        local x,y_prev,y_next = loader:next_batch(split_index)
        x,y_prev,y_next = prepro(x,y_prev,y_next)
        --------------- forward pass encoder ---------------
        local encoder_forw_state = {[0] = encoder_forw_init_state}
        local encoder_back_state = {[0] = encoder_back_init_state}
        for t=1,opt.seq_length do
            clones.encoder_forw[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
            local lst = clones.encoder_forw[t]:forward{x[t], unpack(encoder_forw_state[t-1])}
            encoder_forw_state[t] = {}
            for i=1,#encoder_forw_init_state do table.insert(encoder_forw_state[t], lst[i]) end -- extract the state, without output
        end
        for t=1,opt.seq_length do
            clones.encoder_back[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
            local lst = clones.encoder_back[t]:forward{x[opt.seq_length-t+1], unpack(encoder_back_state[t-1])}
            encoder_back_state[t] = {}
            for i=1,#encoder_back_init_state do table.insert(encoder_back_state[t], lst[i]) end -- extract the state, without output
        end
        local sent_vec = torch.cat(encoder_forw_state[opt.seq_length][1], encoder_back_state[opt.seq_length][1])
        --------------- forward pass decoder ---------------
        local decoder_prev_state = {[0] = decoder_prev_init_state}
        local decoder_next_state = {[0] = decoder_next_init_state}
        local predictions_prev = {}           -- softmax outputs
        local predictions_next = {}           -- softmax outputs
        local loss_prev = 0
        local loss_next = 0
        for t=1,opt.seq_length do
            clones.decoder_prev[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
            local lst = clones.decoder_prev[t]:forward{sent_vec, x[t], unpack(decoder_prev_state[t-1])}
            decoder_prev_state[t] = {}
            for i=1,#decoder_prev_init_state do table.insert(decoder_prev_state[t], lst[i]) end -- extract the state, without output
            predictions_prev[t] = lst[#lst] -- last element is the prediction
            loss_prev = loss_prev + clones.criterion_prev[t]:forward(predictions_prev[t], y_prev[t])
        end
        loss_prev = loss_prev / opt.seq_length
        for t=1,opt.seq_length do
            clones.decoder_next[t]:evaluate() -- make sure we are in correct mode (this is cheap, sets flag)
            local lst = clones.decoder_next[t]:forward{sent_vec, x[t], unpack(decoder_next_state[t-1])}
            decoder_next_state[t] = {}
            for i=1,#decoder_next_init_state do table.insert(decoder_next_state[t], lst[i]) end -- extract the state, without output
            predictions_next[t] = lst[#lst] -- last element is the prediction
            loss_next = loss_next + clones.criterion_next[t]:forward(predictions_next[t], y_next[t])
        end
        loss_next = loss_next / opt.seq_length
        loss = loss + (loss_prev+loss_next)/2
    end

    return loss /n
end

-- start optimization here
local loss0 = nil
train_losses = {}
val_losses = {}
local iterations = opt.max_epochs * loader.ntrain
lr = opt.learning_rate -- starting learning rate which will be decayed
for i = 1, iterations do
    local epoch = i / loader.ntrain
    local timer = torch.Timer()

    local train_loss = myfeval()
    train_losses[i] = train_loss
    local time = timer:time().real

    -- exponential learning rate decay
    if i % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            lr = lr * opt.learning_rate_decay
            print('decayed learning rate by a factor ' .. opt.learning_rate_decay .. ' to ' .. lr)
        end
    end

    -- every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation
        val_losses[i] = val_loss

        local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.vocab_mapping
        torch.save(savefile, checkpoint)
    end

    if i % opt.print_every == 0 then
        grad_norm = encoder_forw_grad_params:norm() / encoder_forw_params:norm()
        grad_norm = grad_norm + encoder_back_grad_params:norm() / encoder_back_params:norm()
        grad_norm = grad_norm + decoder_prev_grad_params:norm() / decoder_prev_params:norm()
        grad_norm = grad_norm + decoder_next_grad_params:norm() / decoder_next_params:norm()
        grad_norm = grad_norm / 4;
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, train_loss, grad_norm, time))
    end

    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss0 == nil then loss0 = train_loss end
    if train_loss > loss0 * 3 then
        print('loss is exploding, aborting.')
        break -- halt
    end
end

