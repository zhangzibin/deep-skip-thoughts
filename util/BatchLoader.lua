local stringx = require('pl.stringx')
local BatchLoader = {}
BatchLoader.__index = BatchLoader

function BatchLoader.create(data_dir, batch_size, seq_length, split_fractions, min_freq)
    local self = {}
    setmetatable(self, BatchLoader)
    local input_file = path.join(data_dir, 'input.txt')
    local vocab_file = path.join(data_dir, 'vocab.t7')
    local tensor_file = path.join(data_dir, 'data.t7')
    BatchLoader.text_to_tensor(input_file, vocab_file, tensor_file, seq_length, min_freq)
    print('loading data files...')
    local data = torch.load(tensor_file)
    self.vocab_mapping = torch.load(vocab_file)
    -- count vocab
    self.vocab_size = 0
    for _ in pairs(self.vocab_mapping) do 
        self.vocab_size = self.vocab_size + 1 
    end
    -- cut off the end so that it divides evenly
    self.batch_size = batch_size
    self.seq_length = seq_length
    local len = data:size(1)
    if len % batch_size ~= 0 then
        data = data:sub(1, batch_size * math.floor(len / batch_size))
    end
    -- make target data, prev & next sentences
    local data_next = data:clone()
    data_next:sub(1,-2):copy(data:sub(2,-1))
    data_next[-1] = data[1]
    local data_prev = data:clone()
    data_prev:sub(2,-1):copy(data:sub(1,-2))
    data_prev[1] = data[-1]
    -- split into batches
    self.x_batches = data:split(batch_size, 1)
    self.ynext_batches = data_next:split(batch_size, 1)
    self.yprev_batches = data_prev:split(batch_size, 1)
    self.nbatches = #self.x_batches
    -- perform safety checks on split_fractions
    assert(split_fractions[1] >= 0 and split_fractions[1] <= 1, 'bad split fraction ' .. split_fractions[1] .. ' for train, not between 0 and 1')
    assert(split_fractions[2] >= 0 and split_fractions[2] <= 1, 'bad split fraction ' .. split_fractions[2] .. ' for val, not between 0 and 1')
    self.ntrain = math.floor(self.nbatches * split_fractions[1])
    self.nval = self.nbatches - self.ntrain
    self.split_sizes = {self.ntrain, self.nval}
    self.batch_ix = {0, 0} -- reset batch pointer
    -- done
    print(string.format('data load done. Number of data batches in train: %d, val: %d', self.ntrain, self.nval))
    collectgarbage()
    return self
end

function BatchLoader.text_to_tensor(in_textfile, out_vocabfile, out_tensorfile, seq_length, min_freq)
    print('creating vocabulary mapping...')
    local f = assert(io.open(in_textfile, "r"))
    local unordered = {}
    local doc_num = 0
    while true do
        local line = f:read("*line")
        if line == nil then break end
        doc_num = doc_num + 1
        for _,word in ipairs(stringx.split(line, ' ')) do
            if not unordered[word] then unordered[word] = 0 end
            unordered[word] = unordered[word] + 1
        end
    end
    f:close()
    -- sort into a table (i.e. keys become 1..N)
    local ordered = {}
    for word,cnt in pairs(unordered) do 
        if cnt > min_freq then ordered[#ordered + 1] = word end
    end
    ordered[#ordered + 1] = '<UNK>'
    ordered[#ordered + 1] = '<EOS>'
    table.sort(ordered)
    -- invert `ordered` to create the char->int mapping
    local vocab_mapping = {}
    for i,word in ipairs(ordered) do
        vocab_mapping[word] = i
    end
    -- construct a tensor with all the data
    print('putting data into tensor...')
    local output_tensor = torch.zeros(doc_num, seq_length):long()
    local f = assert(io.open(in_textfile, "r"))
    local idx = 1
    while true do
        local line = f:read("*line")
        if line == nil then break end
        local jdx = 1
        for _,word in pairs(stringx.split(line, ' ')) do
            if jdx == seq_length then break end
            if vocab_mapping[word] == nil then
                output_tensor[idx][jdx] = vocab_mapping['<UNK>']
            else
                output_tensor[idx][jdx] = vocab_mapping[word]
            end
            jdx = jdx + 1
        end
        for jjdx=jdx,seq_length do
            output_tensor[idx][jjdx] = vocab_mapping['<EOS>']
        end
        idx = idx + 1
    end
    f:close()
    -- save output preprocessed files
    print('saving ' .. out_vocabfile)
    torch.save(out_vocabfile, vocab_mapping)
    print('saving ' .. out_tensorfile)
    torch.save(out_tensorfile, output_tensor)
end

function BatchLoader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end

function BatchLoader:next_batch(split_index)
    if self.split_sizes[split_index] == 0 then
        -- perform a check here to make sure the user isn't screwing something up
        local split_names = {'train', 'val'}
        print('ERROR. Code requested a batch for split ' .. split_names[split_index] .. ', but this split has no data.')
        os.exit() -- crash violently
    end
    -- split_index is integer: 1 = train, 2 = val
    self.batch_ix[split_index] = self.batch_ix[split_index] + 1
    if self.batch_ix[split_index] > self.split_sizes[split_index] then
        self.batch_ix[split_index] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local ix = self.batch_ix[split_index]
    if split_index == 2 then ix = ix + self.ntrain end -- offset by train set size
    return self.x_batches[ix], self.yprev_batches[ix], self.ynext_batches[ix]
end

return BatchLoader
