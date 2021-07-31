# functions definition

function continued_fraction(x, terms)
    val = x
    for i = 1:terms
        if i==terms
            val = 1/(1-val)
        else
            val = x/(1-val)
        end
    end
    return val
   end
   
   
function continued_exp(x, terms)
    val = 1
    for i = 1:terms
        val = exp(val*x)
    end
    return val
end


# continued exponential call section

N = 200
conv = 100
df = 0.01
s = zeros(N, N)
pt = zeros(Complex{Float64},conv)
for x = 1:N , y = 1:N
  dx = df * x - N * df/2
  dy = df * y - N * df/2
  for i = 1:conv
    pt[i] = continued_fraction(dx+dy*im,i)
  end
  s[x,y] = length(Set(pt))
end


using ImageView
using Images
ImageView.imshow(s)


# continued fraction call section

N = 2000
conv = 100
df = 0.01
s = zeros(N, N)
pt = zeros(Complex{Float64},conv)
for x = 1:N , y = 1:N
  dx = df * x - N * df/2
  dy = df * y - N * df/2
  for i = 1:conv
    pt[i] = continued_exp(dx+dy*im,i)
  end
  s[x,y] = length(Set(pt))
end


using ImageView
using Images
ImageView.imshow(s)