### step1
[安装](https://www.jianshu.com/p/8e13c5c48dc5) protoc、grpc for python andd go

### step2
`python -m grpc_tools.protoc --python_out=. --grpc_python_out=. -I. StateAndReward.proto `   
`protoc --go_out=. --go-grpc_out=. product.proto`