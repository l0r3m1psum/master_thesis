#import <XCTest/XCTest.h>

@interface testerino : XCTestCase

@end

@implementation testerino

- (void)setUp {
    // Put setup code here. This method is called before the invocation of each test method in the class.
}

- (void)tearDown {
    // Put teardown code here. This method is called after the invocation of each test method in the class.
}

- (void)testExample {
    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
	XCTAssertTrue(1+1 == 2);
}

- (void)testExample2 {
	// This is an example of a functional test case.
	// Use XCTAssert and related functions to verify your tests produce the correct results.
	XCTAssertLessThan(1, 2);
}

- (void)testPerformanceExample {
    // This is an example of a performance test case.
    [self measureBlock:^{
        // Put the code you want to measure the time of here.
    }];
}

@end
