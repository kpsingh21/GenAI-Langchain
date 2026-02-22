from pydantic import BaseModel, EmailStr, Field

class Student(BaseModel):

    # name: str  ( we can also set default values to )
     name: str = 'nitish-default'
     email: EmailStr
     cgpa: float = Field(gt=0, lt=10, default=5, description='A decimal value representing the cgpa of the student')

    #  age: Optional[int] = None


# new_student = {'name':'nitish'}
new_student={}
# new_student = {'name':32} //  Input should be a valid string [type=string_type, input_value=32, input_type=int]


student = Student(**new_student)

# the object form here is a pydantic object which can be converted into python dictionary
student_dict =dict(student)

print(student)