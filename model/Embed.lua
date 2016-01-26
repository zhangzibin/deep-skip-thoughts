local Embed = {}

function Embed.embed(dim_word, vocab_size)
    local input = nn.Identity()()
    local output = nn.LookupTable(vocab_size, dim_word)(input)
    return nn.gModule({input}, {output})
end

return Embed
