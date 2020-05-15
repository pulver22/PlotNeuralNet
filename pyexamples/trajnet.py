import sys
sys.path.append('../')
from pycore.tikzeng import *

arch = [
    to_head( '../' ),
    to_cor(),
    to_begin(),

    to_input(name="image", pathfile="../examples/images/thorvald.jpg", to="(-1, 0,0)"), 

    # ZEROPADDING
    to_Padding("pad1", offset="(2.5,0,0)", to="(0,0,0)", height=41, depth=41, width=1, opacity=0.5), 
    # BLOCK1
    to_Conv("conv1", "224", 64, offset="(0.5,0,0)", to="(pad1-east)", height=40, depth=40, width=2 ),
    to_BatchNorm("bn_conv1", offset="(0,0,0)", to="(conv1-east)", height=32, depth=32, width=1),
    to_Activation("relu1", offset="(0,0,0)", to="(bn_conv1-east)", height=32, depth=32, width=0.5, opacity=0.2),
    to_Pool("pool1", offset="(0,0,0)", to="(relu1-east)", height=32, depth=32, width=1, opacity=0.5),

    #BLOCK2
    to_Conv("conv2", "55", 48, offset="(2,0,0)", to="(pool1-east)", height=25, depth=25, width=2),
    to_BatchNorm("bn_conv2", offset="(0,0,0)", to="(conv2-east)", height=20, depth=20, width=1),
    to_Activation("relu2", offset="(0,0,0)", to="(bn_conv2-east)", height=20, depth=20, width=0.5, opacity=0.2),
    to_Pool("pool2", offset="(0,0,0)", to="(relu2-east)", height=20, depth=20, width=1, opacity=0.5),

    #BLOCK3
    to_Conv("conv3", "12", 32, offset="(2,0,0)", to="(pool2-east)", height=16, depth=16, width=2),
    to_BatchNorm("bn_conv3", offset="(0,0,0)", to="(conv3-east)", height=12, depth=12, width=1),
    to_Activation("relu3", offset="(0,0,0)", to="(bn_conv3-east)", height=12, depth=12, width=0.5, opacity=0.2),       
    to_Pool("pool3", offset="(0,0,0)", to="(relu3-east)", height=12, depth=12, width=1, opacity=0.5),

    #FLATTEN
    to_FC("flatten", units=1024, offset="(2,0,0)", to="(pool3-east)", height=2, depth=50, width=2),

    #SUM WITH THE INPUTS
    to_add("sum1", offset="(2,0,0)", to="(flatten-east)"),
    # ADDITIONAL INPUT THAT GOES INTO SUM   
    to_FC("scalar_input",units=5, offset="(0.5,-3,0)", to="(flatten-south)", height=2, depth=3, width=2, caption="GoalInfo"),   
    
    # MLP
    to_FC("fc128", units=128, offset="(1,0,0)", to="(sum1-east)", height=2, depth=30, width=2),
    to_Activation("relu4", offset="(0,0,0)", to="(fc128-east)", height=2, depth=30, width=0.5, opacity=0.5),  

    #OUTPUT
    to_FC("output", units=16, offset="(1,0,0)", to="(relu4-east)", height=2, depth=10, width=2),
    to_Activation("tanh", offset="(0,0,0)", to="(output-east)", height=2, depth=10, width=0.5, opacity=0.5),  

    to_connection( "pad1", "conv1"),     
    to_connection( "pool1", "conv2"), 
    to_connection( "pool2", "conv3"),
    to_connection( "conv3", "flatten"), 
    to_connection( "flatten", "sum1"), 
    to_connection( "sum1", "fc128"), 
    to_connection( "relu4", "output"), 
    to_skip_bottom( of='scalar_input', to='sum1', pos=2.25),


    to_end()
    ]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
