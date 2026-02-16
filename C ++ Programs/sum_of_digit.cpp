#include<bits/stdc++.h>
using namespace std;

int main()
{
    int digit,rem,temp,sum=0;
    cout<<"Enter a number: ";
    cin>>digit;

    temp=digit;
    while(temp!=0){
        rem=temp%10;
        sum+=rem;
        temp/=10;
    }

    cout<<"The sum of digits of the number "<<digit<<" is " <<sum<<endl;


    return 0;
}
